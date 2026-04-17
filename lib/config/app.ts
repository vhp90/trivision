import type { UserProfile } from '@/lib/db/types';

export const runtimeDefaults = {
  engineVersion: 'v2.4.1-engine_release',
  verifiedSessionLabel: 'SESSION: VERIFIED',
  localRegionLabel: 'LOCAL',
  localLatencyLabel: '1ms',
  authFooterLabel: 'LOCAL AUTH // ACTIVE',
} as const;

export const authVisualDiagnostics = [
  {
    label: 'DATA_STREAM_ACTIVE //',
    value: 'LOCAL_DB_CONNECTED',
  },
  {
    label: 'LATENCY:',
    value: runtimeDefaults.localLatencyLabel,
  },
  {
    label: 'RUNTIME:',
    value: 'LOCAL_SQLITE',
  },
] as const;

export const generationRuntimeDefaults = {
  outputFormat: 'glb',
  pollIntervalMs: 3000,
  viewerExposure: '1',
  viewerEnvironmentImage: 'neutral',
} as const;

export const generationProviderConfig = {
  lightningTrellisApiUrl: process.env.LIGHTNING_TRELLIS_API_URL?.trim() || '',
} as const;

export const mobileSamRuntimeDefaults = {
  encoderModelPath: '/models/mobilesam.encoder.onnx',
  decoderModelPath: '/models/mobilesam.decoder.quant.onnx',
  wasmBasePath: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/',
  targetLongestSide: 1024,
  maxUploadBytes: 10 * 1024 * 1024,
} as const;

export const studioDefaults = {
  emptyPrompt: '',
  emptyModelName: 'NEW_GENERATION',
  emptyAutoSaveLabel: 'Draft not saved yet',
  emptyMetrics: {
    tris: '--',
    verts: '--',
    fps: '--',
  },
  jobStatusLabels: {
    queued: 'Queued',
    running: 'Generating',
    succeeded: 'Ready',
    failed: 'Failed',
  },
} as const;

export const projectRecordDefaults = {
  updatedLabel: 'Just now',
  trisLabel: 'Pending',
  triCount: '--',
  vertCount: '--',
  fps: '60',
  autoSaveLabel: 'Auto-saved moments ago',
  workspaceUpdatedLabel: 'Updated just now',
} as const;

export const newUserProfileDefaults: Pick<
  UserProfile,
  'roleLabel' | 'region' | 'latencyLabel' | 'sessionLabel' | 'engineVersion' | 'unreadNotifications'
> = {
  roleLabel: 'Studio Member',
  region: runtimeDefaults.localRegionLabel,
  latencyLabel: runtimeDefaults.localLatencyLabel,
  sessionLabel: runtimeDefaults.verifiedSessionLabel,
  engineVersion: runtimeDefaults.engineVersion,
  unreadNotifications: 0,
};

export function createStudioSeed() {
  return String(Math.floor(Math.random() * 900000000) + 100000000);
}
