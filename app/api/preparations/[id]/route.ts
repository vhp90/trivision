import { NextResponse } from 'next/server';
import { getAuthenticatedUser } from '@/lib/auth/session';
import { getAssetPreparationJobForUser } from '@/lib/db/repository';
import {
  getAssetPreparationAssets,
  processAssetPreparationJob,
} from '@/lib/generation/preparation/service';

export const runtime = 'nodejs';
export const maxDuration = 60;

type PreparationStatusRouteProps = {
  params: Promise<{
    id: string;
  }>;
};

export async function GET(_: Request, { params }: PreparationStatusRouteProps) {
  const user = await getAuthenticatedUser();

  if (!user) {
    return NextResponse.json({ message: 'Unauthorized.' }, { status: 401 });
  }

  const routeParams = await params;
  const job = await getAssetPreparationJobForUser(user.id, routeParams.id);

  if (!job) {
    return NextResponse.json({ message: 'Preparation job not found.' }, { status: 404 });
  }

  if (job.status === 'queued' || job.status === 'running') {
    await processAssetPreparationJob(job.id);
  }

  const refreshedJob = await getAssetPreparationJobForUser(user.id, routeParams.id);

  if (!refreshedJob) {
    return NextResponse.json({ message: 'Preparation job not found.' }, { status: 404 });
  }

  return NextResponse.json({
    job: refreshedJob,
    assets: getAssetPreparationAssets(refreshedJob),
  });
}
