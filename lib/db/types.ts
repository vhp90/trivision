import type {
  GenerationJobRecord,
  GenerationJobStatus,
  GenerationParameterValueMap,
} from '@/lib/generation/types';

export type VisualKind = 'factory' | 'lightbulb' | 'globe' | 'car';

export type UserProfile = {
  id: string;
  fullName: string;
  email: string;
  initials: string;
  roleLabel: string;
  region: string;
  latencyLabel: string;
  sessionLabel: string;
  engineVersion: string;
  unreadNotifications: number;
};

export type WorkspaceSummary = {
  id: string;
  userId: string;
  name: string;
  code: string;
  description: string;
  status: string;
  projectCount: number;
  favoriteCount: number;
  updatedLabel: string;
  primaryFocus: string;
  secondaryFocus: string;
  isPrimary: boolean;
};

export type ProjectRecord = {
  id: string;
  userId: string;
  workspaceId: string;
  workspaceName: string;
  name: string;
  format: string | null;
  updatedLabel: string;
  trisLabel: string;
  visual: VisualKind;
  prompt: string;
  seed: string;
  resolution: string;
  creativity: number;
  detailLevel: string;
  triCount: string;
  vertCount: string;
  fps: string;
  autoSaveLabel: string;
  isFavorite: boolean;
  isRecent: boolean;
  status: GenerationJobStatus;
  providerId: string | null;
  modelId: string | null;
  generationJobId: string | null;
  parameterValues: GenerationParameterValueMap;
  sourceImagePath: string | null;
  maskImagePath: string | null;
  outputAssetPath: string | null;
  outputFormat: string | null;
  errorMessage: string | null;
  submittedAt: string | null;
  completedAt: string | null;
};

export type SettingItem = {
  id: string;
  key?: string;
  label: string;
  value: string;
  description: string;
};

export type SettingSection = {
  id: string;
  title: string;
  description: string;
  items: SettingItem[];
};

export type ShellSummary = {
  user: UserProfile;
};

export type SignupPayload = {
  fullName: string;
  email: string;
  password: string;
};

export type LoginPayload = {
  email: string;
  password: string;
};

export type GenerationJobSummary = GenerationJobRecord & {
  projectName: string;
};

export type AssetPreparationStatus = 'queued' | 'running' | 'succeeded' | 'failed';
export type AssetPreparationMode = 'upload' | 'text';
export type AssetPreparationStage = 'text_to_image' | 'remove_background' | 'complete';

export type AssetPreparationJob = {
  id: string;
  userId: string;
  targetModelId: string;
  status: AssetPreparationStatus;
  mode: AssetPreparationMode;
  removeBackground: boolean;
  prompt: string;
  sourceImagePath: string | null;
  generatedImagePath: string | null;
  preparedImagePath: string | null;
  currentStage: AssetPreparationStage;
  fluxTaskId: string | null;
  rmbgTaskId: string | null;
  responsePayloadJson: string | null;
  errorMessage: string | null;
  createdAt: string;
  updatedAt: string;
  completedAt: string | null;
};
