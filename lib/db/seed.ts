import { generationRuntimeDefaults, projectRecordDefaults, runtimeDefaults } from '@/lib/config/app';
import type {
  ProjectRecord,
  SettingSection,
  UserProfile,
  WorkspaceSummary,
} from '@/lib/db/types';
import { getDefaultGenerationModel, getModelParameterDefaults } from '@/lib/generation/registry';

const defaultModel = getDefaultGenerationModel();
const defaultParameters = getModelParameterDefaults(defaultModel);

function createDemoProject(input: Partial<ProjectRecord> & Pick<ProjectRecord, 'id' | 'workspaceId' | 'workspaceName' | 'name' | 'prompt' | 'visual'>): ProjectRecord {
  return {
    id: input.id,
    userId: testAccount.profile.id,
    workspaceId: input.workspaceId,
    workspaceName: input.workspaceName,
    name: input.name,
    format: input.format ?? 'GLB',
    updatedLabel: input.updatedLabel ?? '2h ago',
    trisLabel: input.trisLabel ?? '3D asset ready',
    visual: input.visual,
    prompt: input.prompt,
    seed: input.seed ?? String(defaultParameters.seed ?? 42),
    resolution: input.resolution ?? '1024',
    creativity: input.creativity ?? 85,
    detailLevel: input.detailLevel ?? 'Configured',
    triCount: input.triCount ?? projectRecordDefaults.triCount,
    vertCount: input.vertCount ?? projectRecordDefaults.vertCount,
    fps: input.fps ?? projectRecordDefaults.fps,
    autoSaveLabel: input.autoSaveLabel ?? 'Saved to workspace',
    isFavorite: input.isFavorite ?? false,
    isRecent: input.isRecent ?? true,
    status: input.status ?? 'succeeded',
    providerId: input.providerId ?? defaultModel.providerId,
    modelId: input.modelId ?? defaultModel.id,
    generationJobId: null,
    parameterValues: input.parameterValues ?? defaultParameters,
    sourceImagePath: null,
    maskImagePath: null,
    outputAssetPath: null,
    outputFormat: input.outputFormat ?? generationRuntimeDefaults.outputFormat,
    errorMessage: null,
    submittedAt: input.submittedAt ?? null,
    completedAt: input.completedAt ?? null,
  };
}

export const testAccount = {
  email: 'technical.artist@trivision.io',
  password: 'Trivision123!',
  profile: {
    id: 'user-technical-artist',
    fullName: 'Technical Artist',
    email: 'technical.artist@trivision.io',
    initials: 'TA',
    roleLabel: 'Lead Technical Artist',
    region: 'US-EAST-1',
    latencyLabel: '12ms',
    sessionLabel: runtimeDefaults.verifiedSessionLabel,
    engineVersion: runtimeDefaults.engineVersion,
    unreadNotifications: 0,
  } satisfies UserProfile,
} as const;

export const demoWorkspaces: WorkspaceSummary[] = [
  {
    id: 'workspace-core',
    userId: testAccount.profile.id,
    name: 'Core Sandbox',
    code: 'CORE',
    description: 'Primary environment for experimental text-to-3D generations and system baselines.',
    status: 'Primary',
    projectCount: 4,
    favoriteCount: 2,
    updatedLabel: 'Updated 8m ago',
    primaryFocus: 'Concept modeling',
    secondaryFocus: 'Rapid iteration',
    isPrimary: true,
  },
  {
    id: 'workspace-retail',
    userId: testAccount.profile.id,
    name: 'Retail Signage',
    code: 'RETL',
    description: 'Asset pipeline focused on props, kiosks, and merchandising fixtures.',
    status: 'Active',
    projectCount: 3,
    favoriteCount: 1,
    updatedLabel: 'Updated 42m ago',
    primaryFocus: 'Prop kits',
    secondaryFocus: 'Brand systems',
    isPrimary: false,
  },
  {
    id: 'workspace-automotive',
    userId: testAccount.profile.id,
    name: 'Automotive Lab',
    code: 'AUTO',
    description: 'Vehicle shells, interior explorations, and mobility concept geometry.',
    status: 'Review',
    projectCount: 2,
    favoriteCount: 1,
    updatedLabel: 'Updated 2h ago',
    primaryFocus: 'Vehicle forms',
    secondaryFocus: 'Surface studies',
    isPrimary: false,
  },
];

export const demoProjects: ProjectRecord[] = [
  createDemoProject({
    id: 'project-cyberpunk-vending-machine',
    workspaceId: 'workspace-core',
    workspaceName: 'Core Sandbox',
    name: 'Cyberpunk Vending Machine',
    prompt: 'Cyberpunk neon vending machine',
    visual: 'factory',
    isFavorite: true,
  }),
  createDemoProject({
    id: 'project-neon-sign-motel',
    workspaceId: 'workspace-retail',
    workspaceName: 'Retail Signage',
    name: "Neon Sign 'Motel'",
    prompt: 'Weathered motel neon sign',
    visual: 'lightbulb',
    format: 'OBJ',
  }),
  createDemoProject({
    id: 'project-sci-fi-helmet-concept',
    workspaceId: 'workspace-core',
    workspaceName: 'Core Sandbox',
    name: 'Sci-Fi Helmet Concept',
    prompt: 'Sci-fi helmet with layered visor system',
    visual: 'globe',
    isFavorite: true,
  }),
  createDemoProject({
    id: 'project-hover-car-chassis',
    workspaceId: 'workspace-automotive',
    workspaceName: 'Automotive Lab',
    name: 'Hover Car Chassis',
    prompt: 'Hover car chassis with aerodynamic panels',
    visual: 'car',
    format: 'FBX',
  }),
  createDemoProject({
    id: 'project-industrial-loader-drone',
    workspaceId: 'workspace-core',
    workspaceName: 'Core Sandbox',
    name: 'Industrial Loader Drone',
    prompt: 'Industrial loader drone with articulated lift arms',
    visual: 'factory',
    isFavorite: true,
    isRecent: false,
  }),
  createDemoProject({
    id: 'project-volumetric-sconce',
    workspaceId: 'workspace-retail',
    workspaceName: 'Retail Signage',
    name: 'Volumetric Wall Sconce',
    prompt: 'Premium wall sconce with layered glass diffusion',
    visual: 'lightbulb',
    format: 'OBJ',
    isRecent: false,
  }),
];

export const defaultSettingSections: SettingSection[] = [
  {
    id: 'preferences',
    title: 'Workspace Preferences',
    description: 'Default values applied when you start a new generation session.',
    items: [
      {
        id: 'default-model',
        label: 'Default Model',
        value: defaultModel.shortLabel,
        description: 'Default provider-backed model for new generations.',
      },
      {
        id: 'default-output-format',
        label: 'Default Output Format',
        value: generationRuntimeDefaults.outputFormat.toUpperCase(),
        description: 'Preferred format when exporting generated assets.',
      },
    ],
  },
  {
    id: 'generation',
    title: 'Generation Defaults',
    description: 'Baseline model settings used to keep new assets consistent.',
    items: [
      {
        id: 'default-resolution',
        label: 'Default Resolution',
        value: String(defaultParameters['settings.resolution'] ?? 1024),
        description: 'Baseline voxel resolution for new generations.',
      },
      {
        id: 'default-texture-size',
        label: 'Default Texture Size',
        value: String(defaultParameters['settings.textureSize'] ?? 2048),
        description: 'Baseline texture size for image-to-3D jobs.',
      },
      {
        id: 'default-decimation',
        label: 'Default Decimation Target',
        value: String(defaultParameters['settings.decimationTarget'] ?? 500000),
        description: 'Baseline mesh simplification target for output assets.',
      },
    ],
  },
];

export function createEmptyWorkspace(user: UserProfile): WorkspaceSummary {
  return {
    id: `workspace-${user.id}`,
    userId: user.id,
    name: `${user.fullName.split(' ')[0]}'s Workspace`,
    code: user.initials,
    description: 'Personal workspace for new asset experiments and iteration.',
    status: 'Primary',
    projectCount: 0,
    favoriteCount: 0,
    updatedLabel: projectRecordDefaults.workspaceUpdatedLabel,
    primaryFocus: 'New account setup',
    secondaryFocus: 'First generation',
    isPrimary: true,
  };
}
