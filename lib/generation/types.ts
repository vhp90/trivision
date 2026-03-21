export type GenerationInputKind = 'image' | 'mask';
export type PromptSupport = 'none' | 'optional' | 'required';
export type GenerationModelAvailability = 'enabled' | 'disabled';
export type GenerationJobStatus = 'queued' | 'running' | 'succeeded' | 'failed';
export type GenerationParameterType = 'number' | 'boolean' | 'select';
export type GenerationParameterValue = boolean | number | string;
export type GenerationParameterValueMap = Record<string, GenerationParameterValue>;

export type GenerationCapability = {
  inputKinds: GenerationInputKind[];
  promptSupport: PromptSupport;
  outputFormats: string[];
};

export type GenerationParameterOption = {
  label: string;
  value: string | number;
};

export type GenerationParameterDefinition = {
  key: string;
  label: string;
  description: string;
  section: string;
  type: GenerationParameterType;
  defaultValue: GenerationParameterValue;
  min?: number;
  max?: number;
  step?: number;
  options?: GenerationParameterOption[];
};

export type GenerationModelDefinition = {
  id: string;
  providerId: string;
  label: string;
  shortLabel: string;
  description: string;
  availability: GenerationModelAvailability;
  disabledReason?: string;
  capabilities: GenerationCapability;
  defaultOutputFormat: string;
  primaryInputLabel: string;
  primaryInputDescription: string;
  promptHelperText: string;
  parameters: GenerationParameterDefinition[];
};

export type GenerationRequestPayload = {
  modelId: string;
  prompt: string;
  outputFormat: string;
  parameterValues: GenerationParameterValueMap;
  sourceProjectId?: string | null;
};

export type GenerationExecutionInput = {
  prompt: string;
  outputFormat: string;
  parameterValues: GenerationParameterValueMap;
  sourceImagePath: string;
  maskImagePath?: string | null;
};

export type Runware3DRequest = {
  taskType: '3dInference';
  taskUUID: string;
  model: string;
  inputs: {
    image: string;
    mask?: string;
  };
  positivePrompt?: string;
  outputFormat?: string;
  seed?: number;
  settings?: Record<string, unknown>;
};

export type NormalizedGenerationResult = {
  providerTaskId: string | null;
  assetUrl: string;
  outputFormat: string;
  responsePayload: unknown;
};

export type ProviderStartResult = {
  status: 'completed' | 'running';
  providerTaskId: string | null;
  rawResponse: unknown;
  result?: NormalizedGenerationResult;
};

export type ProviderPollResult = {
  status: 'completed' | 'running';
  rawResponse: unknown;
  result?: NormalizedGenerationResult;
};

export type ProviderExecutionContext = {
  model: GenerationModelDefinition;
  input: GenerationExecutionInput;
};

export type ProviderAdapter = {
  modelId: string;
  validateInput: (context: ProviderExecutionContext) => void;
  buildRunwareRequest: (context: ProviderExecutionContext, inputImageUuid: string, maskImageUuid?: string | null) => Runware3DRequest;
  startGeneration: (
    context: ProviderExecutionContext & {
      inputImageUuid: string;
      maskImageUuid?: string | null;
    },
  ) => Promise<ProviderStartResult>;
  pollGeneration?: (
    context: ProviderExecutionContext & {
      providerTaskId: string;
    },
  ) => Promise<ProviderPollResult>;
  normalizeResult: (rawResponse: unknown, taskUUID: string) => NormalizedGenerationResult;
  mapError: (error: unknown) => string;
};

export type GenerationJobRecord = {
  id: string;
  projectId: string;
  userId: string;
  providerId: string;
  modelId: string;
  status: GenerationJobStatus;
  providerTaskId: string | null;
  requestPayloadJson: string;
  responsePayloadJson: string | null;
  attemptCount: number;
  errorMessage: string | null;
  createdAt: string;
  updatedAt: string;
  startedAt: string | null;
  completedAt: string | null;
};
