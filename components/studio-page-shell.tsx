'use client';

import dynamic from 'next/dynamic';
import type { ProjectRecord, SettingSection } from '@/lib/db/types';

const StudioPageClient = dynamic(
  () => import('@/components/studio-page-client').then((module) => module.StudioPageClient),
  { ssr: false },
);

type StudioPageShellProps = {
  project: ProjectRecord | null;
  settings: SettingSection[];
};

export function StudioPageShell({ project, settings }: StudioPageShellProps) {
  return <StudioPageClient project={project} settings={settings} />;
}
