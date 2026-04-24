import { buildSettingsObject, createGenerationTaskId, getNumberParameter } from '@/lib/generation/helpers';
import { getFriendlyGenerationError } from '@/lib/generation/errors';
import { findTaskResult, RunwareClient } from '@/lib/generation/runware-client';
import type {
  NormalizedGenerationResult,
  ProviderAdapter,
  ProviderExecutionContext,
  ProviderStartResult,
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

function normalizeRunware3DResult(rawResponse: unknown, taskUUID: string): NormalizedGenerationResult {
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

function buildRequest(context: ProviderExecutionContext, inputImageUuid: string): Runware3DRequest {
  const seed = getNumberParameter(context.input.parameterValues, 'seed');
  const settings = buildSettingsObject(context.input.parameterValues);

  return {
    taskType: '3dInference',
    taskUUID: createGenerationTaskId(),
    model: context.model.id,
    inputs: {
      image: inputImageUuid,
    },
    outputFormat: context.input.outputFormat,
    seed: seed ?? undefined,
    settings: Object.keys(settings).length > 0 ? settings : undefined,
  };
}

async function startGeneration(context: ProviderExecutionContext): Promise<ProviderStartResult> {
  const client = new RunwareClient();
  const inputImageUuid = await client.uploadImage(context.sourceImage.buffer, context.sourceImage.mimeType);
  const request = buildRequest(context, inputImageUuid);
  const rawResponse = await client.request([request]);
  const result = normalizeRunware3DResult(rawResponse, request.taskUUID);

  return {
    status: 'completed',
    providerTaskId: result.providerTaskId,
    rawResponse,
    result,
  };
}

export const trellisAdapter: ProviderAdapter = {
  modelId: 'microsoft:trellis-2@4b',
  validateInput(context) {
    if (context.input.prompt.trim()) {
      throw new Error(context.model.promptHelperText);
    }
  },
  startGeneration,
  normalizeResult: normalizeRunware3DResult,
  mapError(error) {
    return getFriendlyGenerationError(error);
  },
};
