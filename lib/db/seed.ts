import { generationRuntimeDefaults, projectRecordDefaults } from '@/lib/config/app';
import type {
  SettingSection,
  UserProfile,
  WorkspaceSummary,
} from '@/lib/db/types';
import { getDefaultGenerationModel, getModelParameterDefaults } from '@/lib/generation/registry';

const defaultModel = getDefaultGenerationModel();
const defaultParameters = getModelParameterDefaults(defaultModel);

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
