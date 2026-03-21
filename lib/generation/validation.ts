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

    throw new Error(`${definition.label} must be true or false.`);
  }

  if (definition.type === 'number') {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }

    throw new Error(`${definition.label} must be a number.`);
  }

  if (typeof value === 'string' || typeof value === 'number') {
    return value;
  }

  throw new Error(`${definition.label} has an invalid value.`);
}

function validateDefinitionValue(
  definition: GenerationParameterDefinition,
  value: GenerationParameterValue,
) {
  if (definition.type === 'number' && typeof value === 'number') {
    if (definition.min !== undefined && value < definition.min) {
      throw new Error(`${definition.label} must be at least ${definition.min}.`);
    }

    if (definition.max !== undefined && value > definition.max) {
      throw new Error(`${definition.label} must be at most ${definition.max}.`);
    }
  }

  if (definition.type === 'select') {
    const optionValues = definition.options?.map((option) => option.value) ?? [];
    const comparableValue = typeof value === 'boolean' ? String(value) : value;

    if (!optionValues.some((optionValue) => String(optionValue) === String(comparableValue))) {
      throw new Error(`${definition.label} must use one of the supported values.`);
    }
  }
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

  if (model.capabilities.promptSupport === 'required' && !trimmedPrompt) {
    throw new Error('A prompt is required for the selected model.');
  }

  if (model.capabilities.promptSupport === 'none' && trimmedPrompt) {
    throw new Error(model.promptHelperText);
  }

  const parameterValues = normalizeParameterValues(model, payload.parameterValues);
  const outputFormat = payload.outputFormat || model.defaultOutputFormat;

  if (!model.capabilities.outputFormats.includes(outputFormat)) {
    throw new Error('The selected output format is not supported by this model.');
  }

  return {
    model,
    payload: {
      ...payload,
      prompt: trimmedPrompt,
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
    validateDefinitionValue(definition, coerced);
    normalized[definition.key] = coerced;
  }

  return normalized;
}
