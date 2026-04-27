export type GenerationInputKind = 'image' | 'mask';
export type PromptSupport = 'none' | 'optional' | 'required';
export type GenerationModelAvailability = 'enabled' | 'disabled';
export type GenerationJobStatus = 'queued' | 'running' | 'succeeded' | 'failed';
export type GenerationParameterType = 'number' | 'boolean' | 'select';
export type GenerationParameterValue = boolean | number | string;
export type GenerationParameterValueMap = Record<string, GenerationParameterValue>;
export type SegmentationPromptMode = 'positive-point' | 'negative-point' | 'box';

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
  segmentationSupport?: {
    engine: 'mobile-sam';
    required: boolean;
    promptModes: SegmentationPromptMode[];
  };
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

export type GenerationInputAsset = {
  path: string;
  fileName: string;
  buffer: Buffer;
  mimeType: string;
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
  deliveryMethod?: 'async' | 'sync';
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
  sourceImage?: GenerationInputAsset | null;
  maskImage?: GenerationInputAsset | null;
};

export type ProviderPollContext = {
  model: GenerationModelDefinition;
  input: GenerationExecutionInput;
  providerTaskId: string;
};

export type ProviderAdapter = {
  modelId: string;
  inputDelivery: 'buffer' | 'url';
  validateInput: (context: ProviderExecutionContext) => void;
  startGeneration: (context: ProviderExecutionContext) => Promise<ProviderStartResult>;
  pollGeneration?: (context: ProviderPollContext) => Promise<ProviderPollResult>;
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
