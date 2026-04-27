import {
  buildSettingsObject,
  createGenerationTaskId,
  getNumberParameter,
} from '@/lib/generation/helpers';
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
  const seed = getNumberParameter(context.input.parameterValues, 'seed');
  const numberResults = getNumberParameter(context.input.parameterValues, 'numberResults');
  const settings = buildSettingsObject(context.input.parameterValues);

  return {
    taskType: '3dInference',
    taskUUID: createGenerationTaskId(),
    model: context.model.id,
    inputs: {
      image: inputImageUuid,
    },
    outputFormat: context.input.outputFormat.toUpperCase(),
    seed: seed ?? undefined,
    numberResults: numberResults ?? undefined,
    settings: Object.keys(settings).length > 0 ? settings : undefined,
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
