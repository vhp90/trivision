import path from 'node:path';
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
  ProviderStartResult,
} from '@/lib/generation/types';

type LightningGeneratePayload = {
  id?: string;
  model_id?: string;
  pipeline_type?: string;
  seed?: number;
  elapsed_seconds?: number;
  download_url?: string;
  filename?: string;
  num_vertices?: number;
  num_faces?: number;
};

function normalizeLightningResult(rawResponse: unknown): NormalizedGenerationResult {
  if (!rawResponse || typeof rawResponse !== 'object') {
    throw new Error('Lightning TRELLIS returned an invalid response.');
  }

  const payload = rawResponse as LightningGeneratePayload;

  if (!payload.download_url) {
    throw new Error('Lightning TRELLIS returned a response without a downloadable asset URL.');
  }

  const providerTaskId = payload.id ?? null;
  const outputFormat = payload.filename
    ? path.extname(payload.filename).replace(/^\./, '') || 'glb'
    : 'glb';

  return {
    providerTaskId,
    assetUrl: payload.download_url,
    outputFormat,
    responsePayload: rawResponse,
  };
}

function buildLightningParameters(context: ProviderExecutionContext) {
  return {
    seed: getNumberParameter(context.input.parameterValues, 'seed') ?? 42,
    pipeline_type: getStringParameter(context.input.parameterValues, 'pipelineType') ?? '1024_cascade',
    preprocess_image: false,
    num_samples: getNumberParameter(context.input.parameterValues, 'numSamples') ?? 1,
    max_num_tokens: getNumberParameter(context.input.parameterValues, 'maxNumTokens') ?? 49152,
    simplify_target: getNumberParameter(context.input.parameterValues, 'simplifyTarget') ?? 1000000,
    texture_size: getNumberParameter(context.input.parameterValues, 'textureSize') ?? 2048,
    remesh: getBooleanParameter(context.input.parameterValues, 'remesh') ?? true,
    remesh_band: getNumberParameter(context.input.parameterValues, 'remeshBand') ?? 1,
    remesh_project: getNumberParameter(context.input.parameterValues, 'remeshProject') ?? 0,
  };
}

async function startGeneration(context: ProviderExecutionContext): Promise<ProviderStartResult> {
  const client = new LightningTrellisClient();
  const rawResponse = await client.generate({
    fileName: context.sourceImage.fileName,
    buffer: context.sourceImage.buffer,
    mimeType: context.sourceImage.mimeType,
    parameters: buildLightningParameters(context),
  });

  const result = normalizeLightningResult(rawResponse);
  const resolvedResult = {
    ...result,
    assetUrl: client.resolveUrl(result.assetUrl),
  };

  return {
    status: 'completed',
    providerTaskId: resolvedResult.providerTaskId,
    rawResponse,
    result: resolvedResult,
  };
}

export const lightningTrellisAdapter: ProviderAdapter = {
  modelId: 'lightning:microsoft-trellis-2@4b',
  validateInput(context) {
    if (context.input.prompt.trim()) {
      throw new Error(context.model.promptHelperText);
    }
  },
  startGeneration,
  normalizeResult(rawResponse) {
    return normalizeLightningResult(rawResponse);
  },
  mapError(error) {
    return getFriendlyGenerationError(error);
  },
};
