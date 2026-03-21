import { StudioPageClient } from '@/components/studio-page-client';
import { requireAuthenticatedUser } from '@/lib/auth/session';
import { getProjectById } from '@/lib/db/repository';

export const dynamic = 'force-dynamic';

type StudioPageProps = {
  searchParams: Promise<{
    projectId?: string;
  }>;
};

export default async function StudioPage({ searchParams }: StudioPageProps) {
  const user = await requireAuthenticatedUser();
  const params = await searchParams;
  const project = params.projectId ? await getProjectById(user.id, params.projectId) : null;

  return <StudioPageClient key={project?.id ?? 'new-generation'} project={project} />;
}
