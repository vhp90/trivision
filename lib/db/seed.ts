import { generationRuntimeDefaults, projectRecordDefaults } from '@/lib/config/app';
import type {
  SettingSection,
  UserProfile,
  WorkspaceSummary,
} from '@/lib/db/types';
import { getDefaultGenerationModel } from '@/lib/generation/registry';

const defaultModel = getDefaultGenerationModel();

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
