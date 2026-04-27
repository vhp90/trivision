import { findTaskResult, getTaskStatus, RunwareClient } from '@/lib/generation/runware-client';

export const FLUX_KLEIN_MODEL_ID = 'runware:400@6';
export const RMBG_MODEL_ID = 'bria:2@1';

export type RunwareImageResult = {
  providerTaskId: string;
  imageUrl: string;
  outputFormat: string;
  responsePayload: unknown;
};

export function buildFluxKleinRequest(input: { taskUUID: string; prompt: string }) {
  return {
    taskType: 'imageInference',
    taskUUID: input.taskUUID,
    model: FLUX_KLEIN_MODEL_ID,
    positivePrompt: input.prompt.trim(),
    width: 1024,
    height: 1024,
    outputFormat: 'PNG',
    numberResults: 1,
  };
}

export function buildRmbgRequest(input: { taskUUID: string; image: string }) {
  return {
    taskType: 'removeBackground',
    taskUUID: input.taskUUID,
    model: RMBG_MODEL_ID,
    inputs: {
      image: input.image,
    },
    outputFormat: 'PNG',
  };
}

function extractImageUrl(taskResult: Record<string, unknown>) {
  const nestedOutputs = taskResult.outputs;

  if (nestedOutputs && typeof nestedOutputs === 'object' && !Array.isArray(nestedOutputs)) {
    const files = (nestedOutputs as Record<string, unknown>).files;

    if (Array.isArray(files)) {
      const firstFile = files.find((file) => file && typeof file === 'object') as Record<string, unknown> | undefined;
      const nestedUrl = firstFile?.url;

      if (typeof nestedUrl === 'string' && nestedUrl.length > 0) {
        return nestedUrl;
      }
    }
  }

  const possibleKeys = [
    'imageURL',
    'imageUrl',
    'imageURI',
    'imageUri',
    'outputURL',
    'outputUrl',
    'fileURL',
    'fileUrl',
    'url',
  ];

  for (const key of possibleKeys) {
    const value = taskResult[key];

    if (typeof value === 'string' && value.length > 0) {
      return value;
    }
  }

  return null;
}

export function normalizeRunwareImageResult(rawResponse: unknown, taskUUID: string): RunwareImageResult {
  const taskResult = findTaskResult(rawResponse, taskUUID);

  if (!taskResult) {
    throw new Error('Runware did not return an image preparation task result.');
  }

  const imageUrl = extractImageUrl(taskResult);

  if (!imageUrl) {
    throw new Error('Runware returned an image response without a downloadable image URL.');
  }

  const outputFormat = taskResult.outputFormat ?? taskResult.format ?? 'png';
  const providerTaskId = taskResult.taskUUID ?? taskResult.taskId ?? taskUUID;

  return {
    providerTaskId: typeof providerTaskId === 'string' ? providerTaskId : taskUUID,
    imageUrl,
    outputFormat: typeof outputFormat === 'string' ? outputFormat.toLowerCase() : 'png',
    responsePayload: rawResponse,
  };
}

export async function submitRunwareImageRequest(request: Record<string, unknown>) {
  const taskUUID = typeof request.taskUUID === 'string' ? request.taskUUID : null;

  if (!taskUUID) {
    throw new Error('Runware image preparation request is missing a task id.');
  }

  const client = new RunwareClient();
  const rawResponse = await client.request([
    {
      ...request,
      deliveryMethod: 'async',
    },
  ]);

  return {
    providerTaskId: taskUUID,
    rawResponse,
  };
}

export async function pollRunwareImageRequest(taskUUID: string) {
  const client = new RunwareClient();
  const rawResponse = await client.getResponse(taskUUID);
  const status = getTaskStatus(rawResponse, taskUUID);

  if (status === 'processing' || status === null) {
    return {
      status: 'running' as const,
      rawResponse,
    };
  }

  if (status === 'success' || status === 'completed') {
    return {
      status: 'completed' as const,
      rawResponse,
      result: normalizeRunwareImageResult(rawResponse, taskUUID),
    };
  }

  throw new Error('Runware image preparation failed while processing.');
}
