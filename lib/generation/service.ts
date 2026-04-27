import path from 'node:path';
import { getGenerationModel } from '@/lib/generation/registry';
import { getProviderAdapter } from '@/lib/generation/providers';
import { normalizeGenerationRequest } from '@/lib/generation/validation';
import { generationRuntimeDefaults } from '@/lib/config/app';
import {
  completeGenerationJob,
  createGenerationDraft,
  failGenerationJob,
  getGenerationJobForProcessing,
  getMaskImagePathFromProject,
  getProjectByIdForProcessing,
  getSourceImagePathFromProject,
  incrementGenerationJobAttempt,
  markGenerationJobProviderPending,
  markGenerationJobRunning,
} from '@/lib/db/repository';
import type {
  GenerationInputAsset,
  GenerationParameterValueMap,
  NormalizedGenerationResult,
} from '@/lib/generation/types';
import { RunwareApiError } from '@/lib/generation/runware-client';
import { getPreparedImagePathForGeneration } from '@/lib/generation/preparation/service';
import { readStoredFile, saveRemoteAsset, saveUploadedFile } from '@/lib/storage/blob';

function getFileExtension(fileName: string) {
  return path.extname(fileName).toLowerCase();
}

function getMimeTypeFromExtension(fileName: string, fallbackType: string) {
  const extension = getFileExtension(fileName);

  if (extension === '.png') {
    return 'image/png';
  }

  if (extension === '.jpg' || extension === '.jpeg') {
    return 'image/jpeg';
  }

  if (extension === '.webp') {
    return 'image/webp';
  }

  return fallbackType || 'application/octet-stream';
}

type StartGenerationInput = {
  userId: string;
  modelId: string;
  prompt: string;
  outputFormat: string;
  parameterValues: GenerationParameterValueMap;
  sourceFile?: File | null;
  maskFile?: File | null;
  sourceProjectId?: string | null;
  preparationJobId?: string | null;
};

function isTransientPollTransportError(error: unknown) {
  if (error instanceof RunwareApiError) {
    return Boolean(
      error.status
      && [408, 429, 500, 502, 503, 504].includes(error.status)
      && !error.taskUUID,
    );
  }

  if (!(error instanceof Error)) {
    return false;
  }

  return error.name === 'AbortError'
    || /fetch failed|network|timed out|Unable to download generated asset/i.test(error.message);
}

export async function startGeneration(input: StartGenerationInput) {
  const model = getGenerationModel(input.modelId);

  if (!model) {
    throw new Error('Selected generation model is not configured.');
  }

  let sourceImagePath: string | null = null;
  let maskImagePath: string | null = null;
  let sourceFileName = 'reference-image';

  if (input.preparationJobId) {
    sourceImagePath = await getPreparedImagePathForGeneration({
      userId: input.userId,
      preparationJobId: input.preparationJobId,
    });
    sourceFileName = 'prepared-source.png';
  } else if (input.sourceFile && input.sourceFile.size > 0) {
    const draftProjectId = `upload-${Date.now()}`;
    const fileBuffer = Buffer.from(await input.sourceFile.arrayBuffer());
    sourceFileName = input.sourceFile.name || sourceFileName;
    sourceImagePath = await saveUploadedFile({
      userId: input.userId,
      projectId: draftProjectId,
      kind: 'inputs',
      fileName: sourceFileName,
      content: fileBuffer,
    });
  } else if (input.sourceProjectId) {
    sourceImagePath = await getSourceImagePathFromProject({
      userId: input.userId,
      projectId: input.sourceProjectId,
    });
    maskImagePath = await getMaskImagePathFromProject({
      userId: input.userId,
      projectId: input.sourceProjectId,
    });

    if (sourceImagePath) {
      sourceFileName = path.basename(sourceImagePath);
    }
  }

  if (input.maskFile && input.maskFile.size > 0) {
    const draftProjectId = `mask-${Date.now()}`;
    const fileBuffer = Buffer.from(await input.maskFile.arrayBuffer());
    maskImagePath = await saveUploadedFile({
      userId: input.userId,
      projectId: draftProjectId,
      kind: 'inputs',
      fileName: input.maskFile.name || 'mask.png',
      content: fileBuffer,
    });
  }

  const normalized = normalizeGenerationRequest(
    {
      modelId: input.modelId,
      prompt: input.prompt,
      outputFormat: input.outputFormat,
      parameterValues: input.parameterValues,
      sourceProjectId: input.sourceProjectId,
    },
    {
      hasSourceImage: Boolean(sourceImagePath),
      hasMaskImage: Boolean(maskImagePath),
    },
  );

  if (!sourceImagePath) {
    throw new Error('An image is required for the selected model.');
  }

  const draft = await createGenerationDraft({
    userId: input.userId,
    modelId: normalized.model.id,
    providerId: normalized.model.providerId,
    prompt: normalized.payload.prompt,
    sourceImagePath,
    maskImagePath,
    sourceFileName,
    outputFormat: normalized.payload.outputFormat,
    parameterValues: normalized.payload.parameterValues,
  });

  await processGenerationJob(draft.jobId);

  return {
    jobId: draft.jobId,
    projectId: draft.projectId,
    status: 'running' as const,
  };
}

export async function processGenerationJob(jobId: string) {
  const job = await getGenerationJobForProcessing(jobId);

  if (!job) {
    return;
  }

  const project = await getProjectByIdForProcessing(job.projectId);

  if (!project || !project.modelId || !project.providerId || !project.sourceImagePath || !project.outputFormat) {
    console.error('[generation] Job missing required project metadata', { jobId, projectId: job?.projectId ?? null });
    await failGenerationJob({
      jobId,
      errorMessage: 'Generation job is missing required project metadata.',
    });
    return;
  }

  const model = getGenerationModel(project.modelId);

  if (!model) {
    console.error('[generation] Job references unknown model', { jobId, modelId: project.modelId });
    await failGenerationJob({
      jobId,
      errorMessage: 'Generation job references an unknown model.',
    });
    return;
  }

  const adapter = getProviderAdapter(model.id);

  try {
    await markGenerationJobRunning(jobId);

    const sourceImage = adapter.inputDelivery === 'buffer'
      ? await loadGenerationAsset(project.sourceImagePath, 'image/png')
      : null;
    const maskImage = adapter.inputDelivery === 'buffer' && project.maskImagePath
      ? await loadGenerationAsset(project.maskImagePath, 'image/png')
      : null;

    const executionContext = {
      model,
      input: {
        prompt: project.prompt,
        outputFormat: project.outputFormat,
        parameterValues: project.parameterValues,
        sourceImagePath: project.sourceImagePath,
        maskImagePath: project.maskImagePath,
      },
      sourceImage,
      maskImage,
    };

    adapter.validateInput(executionContext);

    await incrementGenerationJobAttempt(jobId);
    const providerResult = await adapter.startGeneration(executionContext);

    if (providerResult.status === 'running') {
      if (!providerResult.providerTaskId) {
        throw new Error('The provider did not return a task id for async generation.');
      }

      await markGenerationJobProviderPending({
        jobId,
        providerTaskId: providerResult.providerTaskId,
        responsePayloadJson: JSON.stringify(providerResult.rawResponse),
      });
      return;
    }

    if (!providerResult.result) {
      throw new Error('The provider did not return a completed 3D asset.');
    }

    await saveCompletedGenerationResult({
      jobId,
      userId: project.userId,
      projectId: project.id,
      result: providerResult.result,
    });
  } catch (error) {
    console.error('[generation] Job failed', {
      jobId,
      projectId: project.id,
      modelId: project.modelId,
      error,
    });
    await failGenerationJob({
      jobId,
      errorMessage: adapter.mapError(error),
    });
  }
}

export async function pollGenerationJob(jobId: string) {
  const job = await getGenerationJobForProcessing(jobId);

  if (!job || job.status === 'succeeded' || job.status === 'failed') {
    return;
  }

  if (!job.providerTaskId) {
    await processGenerationJob(jobId);
    return;
  }

  const project = await getProjectByIdForProcessing(job.projectId);

  if (!project || !project.modelId || !project.providerId || !project.sourceImagePath || !project.outputFormat) {
    await failGenerationJob({
      jobId,
      providerTaskId: job.providerTaskId,
      errorMessage: 'Generation job is missing required project metadata.',
    });
    return;
  }

  const model = getGenerationModel(project.modelId);

  if (!model) {
    await failGenerationJob({
      jobId,
      providerTaskId: job.providerTaskId,
      errorMessage: 'Generation job references an unknown model.',
    });
    return;
  }

  const adapter = getProviderAdapter(model.id);

  if (!adapter.pollGeneration) {
    await failGenerationJob({
      jobId,
      providerTaskId: job.providerTaskId,
      errorMessage: 'This provider does not support async generation polling.',
    });
    return;
  }

  try {
    const providerResult = await adapter.pollGeneration({
      model,
      input: {
        prompt: project.prompt,
        outputFormat: project.outputFormat,
        parameterValues: project.parameterValues,
        sourceImagePath: project.sourceImagePath,
        maskImagePath: project.maskImagePath,
      },
      providerTaskId: job.providerTaskId,
    });

    if (providerResult.status === 'running') {
      await markGenerationJobProviderPending({
        jobId,
        providerTaskId: job.providerTaskId,
        responsePayloadJson: JSON.stringify(providerResult.rawResponse),
      });
      return;
    }

    if (!providerResult.result) {
      throw new Error('The provider did not return a completed 3D asset.');
    }

    await saveCompletedGenerationResult({
      jobId,
      userId: project.userId,
      projectId: project.id,
      result: providerResult.result,
    });
  } catch (error) {
    if (isTransientPollTransportError(error)) {
      console.warn('[generation] Polling transport failed; keeping job running', {
        jobId,
        projectId: project.id,
        modelId: project.modelId,
        error,
      });
      return;
    }

    console.error('[generation] Polling failed', {
      jobId,
      projectId: project.id,
      modelId: project.modelId,
      error,
    });
    await failGenerationJob({
      jobId,
      providerTaskId: job.providerTaskId,
      errorMessage: adapter.mapError(error),
    });
  }
}

async function saveCompletedGenerationResult(input: {
  jobId: string;
  userId: string;
  projectId: string;
  result: NormalizedGenerationResult;
}) {
  const outputFormat = input.result.outputFormat || generationRuntimeDefaults.outputFormat;
  const outputAssetPath = await saveRemoteAsset({
    userId: input.userId,
    projectId: input.projectId,
    assetUrl: input.result.assetUrl,
    outputFormat,
  });

  await completeGenerationJob({
    jobId: input.jobId,
    providerTaskId: input.result.providerTaskId,
    responsePayloadJson: JSON.stringify(input.result.responsePayload),
    outputAssetPath,
    outputFormat,
  });
}

async function loadGenerationAsset(relativePath: string, fallbackMimeType: string): Promise<GenerationInputAsset> {
  const buffer = await readStoredFile(relativePath);

  return {
    path: relativePath,
    fileName: path.basename(relativePath),
    buffer,
    mimeType: getMimeTypeFromExtension(relativePath, fallbackMimeType),
  };
}
