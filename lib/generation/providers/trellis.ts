import { createGenerationTaskId } from '@/lib/generation/helpers';
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

function buildRequest(context: ProviderExecutionContext, inputImageUuid: string): Runware3DRequest {
  return {
    taskType: '3dInference',
    taskUUID: createGenerationTaskId(),
    model: context.model.id,
    inputs: {
      image: inputImageUuid,
    },
  };
}

async function startGeneration(context: ProviderExecutionContext): Promise<ProviderStartResult> {
  const request = buildRequest(context, context.input.sourceImagePath);

  return submitRunware3DRequest(request);
}

async function pollGeneration(
  context: ProviderPollContext,
): Promise<ProviderPollResult> {
  return pollRunware3DRequest(context.providerTaskId);
}

export const trellisAdapter: ProviderAdapter = {
  modelId: 'microsoft:trellis-2@4b',
  inputDelivery: 'url',
  validateInput() {},
  startGeneration,
  pollGeneration,
  normalizeResult: normalizeRunware3DResult,
  mapError(error) {
    return getFriendlyGenerationError(error);
  },
};
