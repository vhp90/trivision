import { NextResponse } from 'next/server';
import { getAuthenticatedUser } from '@/lib/auth/session';
import { getGenerationJobForUser, getProjectById } from '@/lib/db/repository';
import { pollGenerationJob } from '@/lib/generation/service';

export const runtime = 'nodejs';
export const maxDuration = 60;

type GenerationStatusRouteProps = {
  params: Promise<{
    id: string;
  }>;
};

export async function GET(_: Request, { params }: GenerationStatusRouteProps) {
  const user = await getAuthenticatedUser();

  if (!user) {
    return NextResponse.json({ message: 'Unauthorized.' }, { status: 401 });
  }

  const routeParams = await params;
  const job = await getGenerationJobForUser(user.id, routeParams.id);

  if (!job) {
    return NextResponse.json({ message: 'Generation job not found.' }, { status: 404 });
  }

  if (job.status === 'queued' || job.status === 'running') {
    await pollGenerationJob(job.id);
  }

  const refreshedJob = await getGenerationJobForUser(user.id, routeParams.id);
  const project = refreshedJob ? await getProjectById(user.id, refreshedJob.projectId) : null;

  if (!project) {
    return NextResponse.json({ message: 'Project not found.' }, { status: 404 });
  }

  return NextResponse.json({
    job: refreshedJob ?? job,
    project,
    assets: {
      sourceImageUrl: project.sourceImagePath ? `/api/projects/${project.id}/asset?kind=source` : null,
      maskImageUrl: project.maskImagePath ? `/api/projects/${project.id}/asset?kind=mask` : null,
      outputAssetUrl: project.outputAssetPath ? `/api/projects/${project.id}/asset?kind=output` : null,
    },
  });
}
