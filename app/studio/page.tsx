import { StudioPageShell } from '@/components/studio-page-shell';
import { requireAuthenticatedUser } from '@/lib/auth/session';
import { getProjectById, getSettings } from '@/lib/db/repository';

export const dynamic = 'force-dynamic';

type StudioPageProps = {
  searchParams: Promise<{
    projectId?: string;
  }>;
};

export default async function StudioPage({ searchParams }: StudioPageProps) {
  const user = await requireAuthenticatedUser();
  const params = await searchParams;
  const [project, settings] = await Promise.all([
    params.projectId ? getProjectById(user.id, params.projectId) : Promise.resolve(null),
    getSettings(user.id),
  ]);

  return <StudioPageShell key={project?.id ?? 'new-generation'} project={project} settings={settings} />;
}
