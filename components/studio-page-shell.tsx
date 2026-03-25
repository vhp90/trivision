'use client';

import dynamic from 'next/dynamic';
import type { ProjectRecord } from '@/lib/db/types';

const StudioPageClient = dynamic(
  () => import('@/components/studio-page-client').then((module) => module.StudioPageClient),
  { ssr: false },
);

type StudioPageShellProps = {
  project: ProjectRecord | null;
};

export function StudioPageShell({ project }: StudioPageShellProps) {
  return <StudioPageClient project={project} />;
}
