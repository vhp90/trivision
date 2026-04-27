import { findTaskResult, getTaskStatus, RunwareClient } from '@/lib/generation/runware-client';
import type {
  NormalizedGenerationResult,
  ProviderPollResult,
  Runware3DRequest,
} from '@/lib/generation/types';

function extractAssetUrl(taskResult: Record<string, unknown>) {
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
    'assetURL',
    'assetUrl',
    'fileURL',
    'fileUrl',
    'outputURL',
    'outputUrl',
    'modelURL',
    'modelUrl',
    'glbURL',
    'glbUrl',
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

export function normalizeRunware3DResult(rawResponse: unknown, taskUUID: string): NormalizedGenerationResult {
  const taskResult = findTaskResult(rawResponse, taskUUID);

  if (!taskResult) {
    throw new Error('Runware did not return a 3D generation task result.');
  }

  const assetUrl = extractAssetUrl(taskResult);

  if (!assetUrl) {
    throw new Error('Runware returned a 3D response without a downloadable asset URL.');
  }

  const outputFormat = taskResult.outputFormat ?? taskResult.format ?? 'glb';
  const providerTaskId = taskResult.taskUUID ?? taskResult.taskId ?? taskUUID;

  return {
    providerTaskId: typeof providerTaskId === 'string' ? providerTaskId : taskUUID,
    assetUrl,
    outputFormat: typeof outputFormat === 'string' ? outputFormat : 'glb',
    responsePayload: rawResponse,
  };
}

export async function submitRunware3DRequest(request: Runware3DRequest) {
  const client = new RunwareClient();
  const rawResponse = await client.request([
    {
      ...request,
      deliveryMethod: 'async',
    },
  ]);

  return {
    status: 'running' as const,
    providerTaskId: request.taskUUID,
    rawResponse,
  };
}

export async function pollRunware3DRequest(taskUUID: string): Promise<ProviderPollResult> {
  const client = new RunwareClient();
  const rawResponse = await client.getResponse(taskUUID);
  const status = getTaskStatus(rawResponse, taskUUID);

  if (status === 'processing' || status === null) {
    return {
      status: 'running',
      rawResponse,
    };
  }

  if (status === 'success' || status === 'completed') {
    const result = normalizeRunware3DResult(rawResponse, taskUUID);

    return {
      status: 'completed',
      rawResponse,
      result,
    };
  }

  throw new Error('Runware generation failed while processing the 3D asset.');
}
