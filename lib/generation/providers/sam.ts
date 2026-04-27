import { createGenerationTaskId, getNumberParameter } from '@/lib/generation/helpers';
import { getFriendlyGenerationError } from '@/lib/generation/errors';
import {
  normalizeRunware3DResult,
  pollRunware3DRequest,
  submitRunware3DRequest,
} from '@/lib/generation/providers/runware-3d';
import type {
  ProviderAdapter,
  ProviderExecutionContext,
  ProviderPollResult,
  ProviderPollContext,
  ProviderStartResult,
  Runware3DRequest,
} from '@/lib/generation/types';

function buildRequest(
  context: ProviderExecutionContext,
  inputImageUuid: string,
  maskImageUuid?: string | null,
): Runware3DRequest {
  if (!maskImageUuid) {
    throw new Error('SAM 3D requires a mask input.');
  }

  const seed = getNumberParameter(context.input.parameterValues, 'seed');
  const positivePrompt = context.input.prompt.trim();

  return {
    taskType: '3dInference',
    taskUUID: createGenerationTaskId(),
    model: context.model.id,
    inputs: {
      image: inputImageUuid,
      mask: maskImageUuid,
    },
    positivePrompt: positivePrompt || undefined,
    seed: seed ?? undefined,
  };
}

async function startGeneration(
  context: ProviderExecutionContext,
): Promise<ProviderStartResult> {
  const request = buildRequest(context, context.input.sourceImagePath, context.input.maskImagePath);

  return submitRunware3DRequest(request);
}

async function pollGeneration(
  context: ProviderPollContext,
): Promise<ProviderPollResult> {
  return pollRunware3DRequest(context.providerTaskId);
}

export const samAdapter: ProviderAdapter = {
  modelId: 'meta:sam@3d',
  inputDelivery: 'url',
  validateInput(context) {
    if (!context.input.maskImagePath) {
      throw new Error('SAM 3D requires a mask input.');
    }
  },
  startGeneration,
  pollGeneration,
  normalizeResult: normalizeRunware3DResult,
  mapError(error) {
    return getFriendlyGenerationError(error);
  },
};
