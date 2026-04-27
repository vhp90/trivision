import path from 'node:path';
import { randomUUID } from 'node:crypto';
import {
  completeAssetPreparationJob,
  createAssetPreparationJob,
  failAssetPreparationJob,
  getAssetPreparationJobForProcessing,
  getAssetPreparationJobForUser,
  getSourceImagePathFromProject,
  markAssetPreparationRunning,
} from '@/lib/db/repository';
import { getGenerationModel } from '@/lib/generation/registry';
import { getFriendlyGenerationError } from '@/lib/generation/errors';
import { createGenerationTaskId } from '@/lib/generation/helpers';
import {
  buildFluxKleinRequest,
  buildRmbgRequest,
  pollRunwareImageRequest,
  submitRunwareImageRequest,
} from '@/lib/generation/preparation/runware-image';
import { saveRemoteFile, saveUploadedFile } from '@/lib/storage/blob';
import type { AssetPreparationJob } from '@/lib/db/types';

type StartAssetPreparationInput = {
  userId: string;
  targetModelId: string;
  textToImage: boolean;
  removeBackground: boolean;
  prompt: string;
  sourceFile?: File | null;
  sourceProjectId?: string | null;
};

function getPreparedFileName(prefix: string, outputFormat: string) {
  return `${prefix}.${outputFormat.replace(/^\./, '').toLowerCase() || 'png'}`;
}

function getPublicImagePath(job: AssetPreparationJob) {
  return job.preparedImagePath ?? job.generatedImagePath ?? job.sourceImagePath;
}

export function getAssetPreparationAssets(job: AssetPreparationJob) {
  const publicImagePath = getPublicImagePath(job);

  return {
    sourceImageUrl: job.sourceImagePath,
    generatedImageUrl: job.generatedImagePath,
    preparedImageUrl: publicImagePath,
  };
}

export async function getPreparedImagePathForGeneration(input: {
  userId: string;
  preparationJobId: string;
}) {
  const job = await getAssetPreparationJobForUser(input.userId, input.preparationJobId);

  if (!job || job.status !== 'succeeded') {
    throw new Error('Prepared source image is not ready yet.');
  }

  const preparedImagePath = getPublicImagePath(job);

  if (!preparedImagePath) {
    throw new Error('Prepared source image is missing.');
  }

  return preparedImagePath;
}

export async function startAssetPreparation(input: StartAssetPreparationInput) {
  const model = getGenerationModel(input.targetModelId);

  if (!model || model.availability !== 'enabled' || !model.capabilities.inputKinds.includes('image')) {
    throw new Error('Selected model does not support image-based generation.');
  }

  const prompt = input.prompt.trim();

  if (input.textToImage && !prompt) {
    throw new Error('Enter a prompt before generating a source image.');
  }

  if (!input.textToImage && !input.sourceFile && !input.sourceProjectId) {
    throw new Error('Upload a source image before preparing it.');
  }

  if (!input.textToImage && !input.removeBackground) {
    throw new Error('Choose text-to-image or background removal before preparing a source.');
  }

  const jobId = `prep-${Date.now()}-${randomUUID()}`;
  let sourceImagePath: string | null = null;

  if (!input.textToImage && input.sourceFile && input.sourceFile.size > 0) {
    const fileBuffer = Buffer.from(await input.sourceFile.arrayBuffer());
    sourceImagePath = await saveUploadedFile({
      userId: input.userId,
      projectId: jobId,
      kind: 'inputs',
      fileName: input.sourceFile.name || 'source-image.png',
      content: fileBuffer,
    });
  } else if (!input.textToImage && input.sourceProjectId) {
    sourceImagePath = await getSourceImagePathFromProject({
      userId: input.userId,
      projectId: input.sourceProjectId,
    });

    if (!sourceImagePath) {
      throw new Error('Selected project does not have a source image.');
    }
  }

  await createAssetPreparationJob({
    id: jobId,
    userId: input.userId,
    targetModelId: input.targetModelId,
    mode: input.textToImage ? 'text' : 'upload',
    removeBackground: input.removeBackground,
    prompt,
    sourceImagePath,
    currentStage: input.textToImage ? 'text_to_image' : 'remove_background',
  });

  await processAssetPreparationJob(jobId);

  const job = await getAssetPreparationJobForUser(input.userId, jobId);

  if (!job) {
    throw new Error('Prepared source job could not be created.');
  }

  return job;
}

export async function processAssetPreparationJob(jobId: string) {
  const job = await getAssetPreparationJobForProcessing(jobId);

  if (!job || job.status === 'succeeded' || job.status === 'failed') {
    return;
  }

  try {
    if (job.currentStage === 'text_to_image') {
      await processTextToImageStage(job);
      return;
    }

    if (job.currentStage === 'remove_background') {
      await processRemoveBackgroundStage(job);
    }
  } catch (error) {
    await failAssetPreparationJob({
      jobId: job.id,
      errorMessage: getFriendlyGenerationError(error),
    });
  }
}

async function processTextToImageStage(job: AssetPreparationJob) {
  if (!job.fluxTaskId) {
    const taskUUID = createGenerationTaskId();
    const request = buildFluxKleinRequest({
      taskUUID,
      prompt: job.prompt,
    });
    const submitResult = await submitRunwareImageRequest(request);

    await markAssetPreparationRunning({
      jobId: job.id,
      currentStage: 'text_to_image',
      fluxTaskId: submitResult.providerTaskId,
      responsePayloadJson: JSON.stringify(submitResult.rawResponse),
    });
    return;
  }

  const pollResult = await pollRunwareImageRequest(job.fluxTaskId);

  if (pollResult.status === 'running') {
    await markAssetPreparationRunning({
      jobId: job.id,
      currentStage: 'text_to_image',
      responsePayloadJson: JSON.stringify(pollResult.rawResponse),
    });
    return;
  }

  const generatedPath = await saveRemoteFile({
    userId: job.userId,
    projectId: job.id,
    kind: 'inputs',
    assetUrl: pollResult.result.imageUrl,
    fileName: getPreparedFileName('flux-source', pollResult.result.outputFormat),
  });

  if (!job.removeBackground) {
    await completeAssetPreparationJob({
      jobId: job.id,
      generatedImagePath: generatedPath,
      preparedImagePath: generatedPath,
      responsePayloadJson: JSON.stringify(pollResult.rawResponse),
    });
    return;
  }

  const rmbgTaskId = createGenerationTaskId();
  const request = buildRmbgRequest({
    taskUUID: rmbgTaskId,
    image: generatedPath,
  });
  const submitResult = await submitRunwareImageRequest(request);

  await markAssetPreparationRunning({
    jobId: job.id,
    currentStage: 'remove_background',
    generatedImagePath: generatedPath,
    rmbgTaskId: submitResult.providerTaskId,
    responsePayloadJson: JSON.stringify(submitResult.rawResponse),
  });
}

async function processRemoveBackgroundStage(job: AssetPreparationJob) {
  const sourcePath = job.generatedImagePath ?? job.sourceImagePath;

  if (!sourcePath) {
    throw new Error('Background removal needs a source image.');
  }

  if (!job.rmbgTaskId) {
    const taskUUID = createGenerationTaskId();
    const request = buildRmbgRequest({
      taskUUID,
      image: sourcePath,
    });
    const submitResult = await submitRunwareImageRequest(request);

    await markAssetPreparationRunning({
      jobId: job.id,
      currentStage: 'remove_background',
      rmbgTaskId: submitResult.providerTaskId,
      responsePayloadJson: JSON.stringify(submitResult.rawResponse),
    });
    return;
  }

  const pollResult = await pollRunwareImageRequest(job.rmbgTaskId);

  if (pollResult.status === 'running') {
    await markAssetPreparationRunning({
      jobId: job.id,
      currentStage: 'remove_background',
      responsePayloadJson: JSON.stringify(pollResult.rawResponse),
    });
    return;
  }

  const preparedPath = await saveRemoteFile({
    userId: job.userId,
    projectId: job.id,
    kind: 'inputs',
    assetUrl: pollResult.result.imageUrl,
    fileName: getPreparedFileName(`rmbg-${path.basename(sourcePath).replace(/\.[^.]+$/, '')}`, pollResult.result.outputFormat),
  });

  await completeAssetPreparationJob({
    jobId: job.id,
    preparedImagePath: preparedPath,
    responsePayloadJson: JSON.stringify(pollResult.rawResponse),
  });
}
