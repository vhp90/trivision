import { NextResponse } from 'next/server';
import { getAuthenticatedUser } from '@/lib/auth/session';
import { getGenerationJobForUser, getProjectById } from '@/lib/db/repository';

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

  const project = await getProjectById(user.id, job.projectId);

  if (!project) {
    return NextResponse.json({ message: 'Project not found.' }, { status: 404 });
  }

  return NextResponse.json({
    job,
    project,
    assets: {
      sourceImageUrl: project.sourceImagePath ? `/api/projects/${project.id}/asset?kind=source` : null,
      maskImageUrl: project.maskImagePath ? `/api/projects/${project.id}/asset?kind=mask` : null,
      outputAssetUrl: project.outputAssetPath ? `/api/projects/${project.id}/asset?kind=output` : null,
    },
  });
}
