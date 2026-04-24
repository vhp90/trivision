import { NextResponse } from 'next/server';
import { getAuthenticatedUser } from '@/lib/auth/session';
import { getProjectById } from '@/lib/db/repository';
import { getFriendlyGenerationError } from '@/lib/generation/errors';
import { startGeneration } from '@/lib/generation/service';

type ProjectRetryRouteProps = {
  params: Promise<{
    projectId: string;
  }>;
};

export const runtime = 'nodejs';
export const maxDuration = 60;

export async function POST(_: Request, { params }: ProjectRetryRouteProps) {
  const user = await getAuthenticatedUser();

  if (!user) {
    return NextResponse.json({ message: 'Unauthorized.' }, { status: 401 });
  }

  const routeParams = await params;
  const project = await getProjectById(user.id, routeParams.projectId);

  if (!project) {
    return NextResponse.json({ message: 'Generation not found.' }, { status: 404 });
  }

  if (project.status === 'queued' || project.status === 'running') {
    return NextResponse.json({ message: 'This generation is already running.' }, { status: 409 });
  }

  if (!project.modelId || !project.outputFormat || !project.sourceImagePath) {
    return NextResponse.json(
      { message: 'This generation is missing the saved input needed for retry.' },
      { status: 400 },
    );
  }

  try {
    const result = await startGeneration({
      userId: user.id,
      modelId: project.modelId,
      prompt: project.prompt,
      outputFormat: project.outputFormat,
      parameterValues: project.parameterValues,
      sourceProjectId: project.id,
    });

    return NextResponse.json(result, { status: 202 });
  } catch (error) {
    return NextResponse.json(
      { message: getFriendlyGenerationError(error) },
      { status: 400 },
    );
  }
}
