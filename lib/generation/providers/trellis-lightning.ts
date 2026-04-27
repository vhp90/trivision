import {
  getBooleanParameter,
  getNumberParameter,
  getStringParameter,
} from '@/lib/generation/helpers';
import { getFriendlyGenerationError } from '@/lib/generation/errors';
import { LightningTrellisClient } from '@/lib/generation/lightning-client';
import type {
  NormalizedGenerationResult,
  ProviderAdapter,
  ProviderExecutionContext,
  ProviderPollContext,
  ProviderPollResult,
  ProviderStartResult,
} from '@/lib/generation/types';

type LightningSubmitPayload = {
  job_id?: string;
  queue_position?: number;
  status?: string;
};

type LightningStatusPayload = {
  job_id?: string;
  status?: string;
  elapsed_time?: number | null;
  stage_times?: Record<string, number> | null;
};

type LightningParameterContext = Pick<ProviderExecutionContext, 'input'> | {
  parameterValues: ProviderExecutionContext['input']['parameterValues'];
};

function getParameterValues(context: LightningParameterContext) {
  return 'input' in context ? context.input.parameterValues : context.parameterValues;
}

export function normalizeLightningSubmitResult(rawResponse: unknown): ProviderStartResult {
  if (!rawResponse || typeof rawResponse !== 'object') {
    throw new Error('Lightning TRELLIS returned an invalid response.');
  }

  const payload = rawResponse as LightningSubmitPayload;

  if (!payload.job_id) {
    throw new Error('Lightning TRELLIS did not return a job id.');
  }

  return {
    status: 'running',
    providerTaskId: payload.job_id,
    rawResponse,
  };
}

export function normalizeLightningJobResult(rawResponse: unknown, apiUrl: string): NormalizedGenerationResult {
  if (!rawResponse || typeof rawResponse !== 'object') {
    throw new Error('Lightning TRELLIS returned an invalid status response.');
  }

  const payload = rawResponse as LightningStatusPayload;

  if (!payload.job_id) {
    throw new Error('Lightning TRELLIS status response did not include a job id.');
  }

  return {
    providerTaskId: payload.job_id,
    assetUrl: new LightningTrellisClient(apiUrl).getResultUrl(payload.job_id),
    outputFormat: 'glb',
    responsePayload: rawResponse,
  };
}

export function buildLightningParameters(context: LightningParameterContext) {
  const parameterValues = getParameterValues(context);

  return {
    seed: getNumberParameter(parameterValues, 'seed') ?? 42,
    pipeline_type: getStringParameter(parameterValues, 'pipelineType') ?? '512',
    preprocess_image: getBooleanParameter(parameterValues, 'preprocessImage') ?? true,
    decimation_target: getNumberParameter(parameterValues, 'decimationTarget') ?? 1000000,
    texture_size: getNumberParameter(parameterValues, 'textureSize') ?? 4096,
    remesh: getBooleanParameter(parameterValues, 'remesh') ?? true,
    simplify_limit: getNumberParameter(parameterValues, 'simplifyLimit') ?? 16777216,
  };
}

async function startGeneration(context: ProviderExecutionContext): Promise<ProviderStartResult> {
  if (!context.sourceImage) {
    throw new Error('Lightning TRELLIS requires a source image file.');
  }

  const client = new LightningTrellisClient();
  const rawResponse = await client.generate({
    fileName: context.sourceImage.fileName,
    buffer: context.sourceImage.buffer,
    mimeType: context.sourceImage.mimeType,
    parameters: buildLightningParameters(context),
  });

  return normalizeLightningSubmitResult(rawResponse);
}

async function pollGeneration(context: ProviderPollContext): Promise<ProviderPollResult> {
  const client = new LightningTrellisClient();
  const rawResponse = await client.getJobStatus(context.providerTaskId);
  const status = rawResponse.status;

  if (
    status === 'queued'
    || status === 'preprocessing'
    || status === 'sparse_structure'
    || status === 'shape_generation'
    || status === 'texture_generation'
    || status === 'postprocessing'
  ) {
    return {
      status: 'running',
      rawResponse,
    };
  }

  if (status === 'complete') {
    return {
      status: 'completed',
      rawResponse,
      result: normalizeLightningJobResult(rawResponse, client.baseUrl),
    };
  }

  if (status === 'failed' || status === 'cancelled') {
    throw new Error(`Lightning TRELLIS job ${status}.`);
  }

  throw new Error(`Lightning TRELLIS returned an unknown job status: ${status ?? 'missing'}.`);
}

export const lightningTrellisAdapter: ProviderAdapter = {
  modelId: 'lightning:microsoft-trellis-2@4b',
  inputDelivery: 'buffer',
  validateInput(context) {
    if (context.input.prompt.trim()) {
      throw new Error(context.model.promptHelperText);
    }
  },
  startGeneration,
  pollGeneration,
  normalizeResult(rawResponse) {
    return normalizeLightningJobResult(rawResponse, new LightningTrellisClient().baseUrl);
  },
  mapError(error) {
    return getFriendlyGenerationError(error);
  },
};
