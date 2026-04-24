import {
  getGenerationModel,
  getModelParameterDefaults,
} from '@/lib/generation/registry';
import type {
  GenerationModelDefinition,
  GenerationParameterDefinition,
  GenerationParameterValue,
  GenerationParameterValueMap,
  GenerationRequestPayload,
} from '@/lib/generation/types';

function coerceParameterValue(
  definition: GenerationParameterDefinition,
  value: unknown,
): GenerationParameterValue {
  if (definition.type === 'boolean') {
    if (typeof value === 'boolean') {
      return value;
    }

    if (typeof value === 'string') {
      const normalizedValue = value.trim().toLowerCase();

      if (normalizedValue === 'true') {
        return true;
      }

      if (normalizedValue === 'false') {
        return false;
      }
    }

    return definition.defaultValue;
  }

  if (definition.type === 'number') {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }

    if (typeof value === 'string' && value.trim()) {
      const numericValue = Number(value);

      if (Number.isFinite(numericValue)) {
        return numericValue;
      }
    }

    return definition.defaultValue;
  }

  if (typeof value === 'string' || typeof value === 'number') {
    return value;
  }

  return definition.defaultValue;
}

function validateDefinitionValue(
  definition: GenerationParameterDefinition,
  value: GenerationParameterValue,
) {
  let normalizedValue = value;

  if (definition.type === 'number' && typeof value === 'number') {
    if (definition.min !== undefined && value < definition.min) {
      normalizedValue = definition.min;
    }

    if (definition.max !== undefined && typeof normalizedValue === 'number' && normalizedValue > definition.max) {
      normalizedValue = definition.max;
    }
  }

  if (definition.type === 'select') {
    const optionValues = definition.options?.map((option) => option.value) ?? [];
    const comparableValue = typeof normalizedValue === 'boolean' ? String(normalizedValue) : normalizedValue;
    const matchedOptionValue = optionValues.find((optionValue) => String(optionValue) === String(comparableValue));

    if (matchedOptionValue === undefined) {
      normalizedValue = definition.defaultValue;
    } else {
      normalizedValue = matchedOptionValue;
    }
  }

  return normalizedValue;
}

export function normalizeGenerationRequest(
  payload: GenerationRequestPayload,
  options: { hasSourceImage: boolean; hasMaskImage?: boolean },
) {
  const model = getGenerationModel(payload.modelId);

  if (!model) {
    throw new Error('Selected generation model is not configured.');
  }

  if (model.availability !== 'enabled') {
    throw new Error(model.disabledReason ?? 'Selected generation model is not currently available.');
  }

  if (model.capabilities.inputKinds.includes('image') && !options.hasSourceImage) {
    throw new Error('An image is required for the selected model.');
  }

  if (model.capabilities.inputKinds.includes('mask') && !options.hasMaskImage) {
    throw new Error('A mask image is required for the selected model.');
  }

  const trimmedPrompt = payload.prompt.trim();
  const supportedPrompt = model.capabilities.promptSupport === 'none' ? '' : trimmedPrompt;

  if (model.capabilities.promptSupport === 'required' && !supportedPrompt) {
    throw new Error('A prompt is required for the selected model.');
  }

  const parameterValues = normalizeParameterValues(model, payload.parameterValues);
  const outputFormat = (payload.outputFormat || model.defaultOutputFormat).toLowerCase();

  if (!model.capabilities.outputFormats.includes(outputFormat)) {
    throw new Error('The selected output format is not supported by this model.');
  }

  return {
    model,
    payload: {
      ...payload,
      prompt: supportedPrompt,
      outputFormat,
      parameterValues,
    },
  };
}

export function normalizeParameterValues(
  model: GenerationModelDefinition,
  incomingValues: GenerationParameterValueMap,
) {
  const defaults = getModelParameterDefaults(model);
  const normalized: GenerationParameterValueMap = { ...defaults };

  for (const definition of model.parameters) {
    if (!(definition.key in incomingValues)) {
      continue;
    }

    const coerced = coerceParameterValue(definition, incomingValues[definition.key]);
    normalized[definition.key] = validateDefinitionValue(definition, coerced);
  }

  return normalized;
}
