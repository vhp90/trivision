import { readFile } from 'node:fs/promises';
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
  markGenerationJobRunning,
} from '@/lib/db/repository';
import type { GenerationInputAsset, GenerationParameterValueMap } from '@/lib/generation/types';
import { saveRemoteAsset, saveUploadedFile } from '@/lib/storage/local';

const activeJobs = new Map<string, Promise<void>>();

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
};

export async function startGeneration(input: StartGenerationInput) {
  const model = getGenerationModel(input.modelId);

  if (!model) {
    throw new Error('Selected generation model is not configured.');
  }

  let sourceImagePath: string | null = null;
  let maskImagePath: string | null = null;
  let sourceFileName = 'reference-image';

  if (input.sourceFile && input.sourceFile.size > 0) {
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

  const queuedPromise = processGenerationJob(draft.jobId).finally(() => {
    activeJobs.delete(draft.jobId);
  });

  activeJobs.set(draft.jobId, queuedPromise);

  return {
    jobId: draft.jobId,
    projectId: draft.projectId,
    status: 'queued' as const,
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

    const sourceImage = await loadGenerationAsset(project.sourceImagePath, 'image/png');
    const maskImage = project.maskImagePath
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

    const providerResult = await adapter.startGeneration(executionContext);

    if (providerResult.status !== 'completed' || !providerResult.result) {
      throw new Error('The provider did not return a completed 3D asset.');
    }

    const outputAssetPath = await saveRemoteAsset({
      userId: project.userId,
      projectId: project.id,
      assetUrl: providerResult.result.assetUrl,
      outputFormat: providerResult.result.outputFormat || generationRuntimeDefaults.outputFormat,
    });

    await completeGenerationJob({
      jobId,
      providerTaskId: providerResult.result.providerTaskId,
      responsePayloadJson: JSON.stringify(providerResult.result.responsePayload),
      outputAssetPath,
      outputFormat: providerResult.result.outputFormat || generationRuntimeDefaults.outputFormat,
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

async function loadGenerationAsset(relativePath: string, fallbackMimeType: string): Promise<GenerationInputAsset> {
  const absolutePath = path.join(process.cwd(), 'data', 'storage', relativePath);
  const buffer = await readFile(absolutePath);

  return {
    path: relativePath,
    fileName: path.basename(relativePath),
    buffer,
    mimeType: getMimeTypeFromExtension(relativePath, fallbackMimeType),
  };
}
