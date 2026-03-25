import { createGenerationTaskId, getNumberParameter } from '@/lib/generation/helpers';
import { findTaskResult, RunwareClient } from '@/lib/generation/runware-client';
import type {
  NormalizedGenerationResult,
  ProviderAdapter,
  ProviderExecutionContext,
  ProviderStartResult,
  Runware3DRequest,
} from '@/lib/generation/types';

const RUNWARE_BLANK_PROMPT = '__BLANK__';

function normalizeRunware3DResult(rawResponse: unknown, taskUUID: string): NormalizedGenerationResult {
  const taskResult = findTaskResult(rawResponse, taskUUID);

  if (!taskResult) {
    throw new Error('Runware did not return a SAM 3D task result.');
  }

  const nestedOutputs = taskResult.outputs;
  const nestedFiles = nestedOutputs && typeof nestedOutputs === 'object' && !Array.isArray(nestedOutputs)
    ? (nestedOutputs as Record<string, unknown>).files
    : null;
  const nestedUrl = Array.isArray(nestedFiles)
    ? ((nestedFiles.find((file) => file && typeof file === 'object') as Record<string, unknown> | undefined)?.url ?? null)
    : null;
  const assetUrl = nestedUrl
    ?? taskResult.assetURL
    ?? taskResult.assetUrl
    ?? taskResult.fileURL
    ?? taskResult.fileUrl
    ?? taskResult.outputURL
    ?? taskResult.outputUrl
    ?? taskResult.modelURL
    ?? taskResult.modelUrl
    ?? taskResult.glbURL
    ?? taskResult.glbUrl
    ?? taskResult.url;

  if (typeof assetUrl !== 'string') {
    throw new Error('Runware returned a SAM 3D response without a downloadable asset URL.');
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

function buildRequest(
  context: ProviderExecutionContext,
  inputImageUuid: string,
  maskImageUuid?: string | null,
): Runware3DRequest {
  if (!maskImageUuid) {
    throw new Error('SAM 3D requires a mask input.');
  }

  const seed = getNumberParameter(context.input.parameterValues, 'seed');
  const positivePrompt = context.input.prompt.trim() || RUNWARE_BLANK_PROMPT;

  return {
    taskType: '3dInference',
    taskUUID: createGenerationTaskId(),
    model: context.model.id,
    inputs: {
      image: inputImageUuid,
      mask: maskImageUuid,
    },
    positivePrompt,
    outputFormat: context.input.outputFormat,
    seed: seed ?? undefined,
  };
}

async function startGeneration(
  context: ProviderExecutionContext & {
    inputImageUuid: string;
    maskImageUuid?: string | null;
  },
): Promise<ProviderStartResult> {
  const client = new RunwareClient();
  const request = buildRequest(context, context.inputImageUuid, context.maskImageUuid);
  const rawResponse = await client.request([request]);
  const result = normalizeRunware3DResult(rawResponse, request.taskUUID);

  return {
    status: 'completed',
    providerTaskId: result.providerTaskId,
    rawResponse,
    result,
  };
}

export const samAdapter: ProviderAdapter = {
  modelId: 'meta:sam@3d',
  validateInput(context) {
    if (!context.input.maskImagePath) {
      throw new Error('SAM 3D requires a mask input.');
    }
  },
  buildRunwareRequest(context, inputImageUuid, maskImageUuid) {
    return buildRequest(context, inputImageUuid, maskImageUuid);
  },
  startGeneration,
  normalizeResult: normalizeRunware3DResult,
  mapError(error) {
    return error instanceof Error ? error.message : 'SAM 3D generation failed.';
  },
};
