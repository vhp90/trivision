import { NextResponse } from 'next/server';
import { getAuthenticatedUser } from '@/lib/auth/session';
import {
  deleteProjectForUser,
  getProjectAssetReferenceCounts,
  updateProjectForUser,
} from '@/lib/db/repository';
import { normalizeProjectUpdateInput } from '@/lib/db/project-actions';
import { deleteStoredFile } from '@/lib/storage/local';

type ProjectRouteProps = {
  params: Promise<{
    projectId: string;
  }>;
};

export async function PATCH(request: Request, { params }: ProjectRouteProps) {
  const user = await getAuthenticatedUser();

  if (!user) {
    return NextResponse.json({ message: 'Unauthorized.' }, { status: 401 });
  }

  const body = await request.json().catch(() => null) as {
    name?: unknown;
    isFavorite?: unknown;
  } | null;
  const updates = normalizeProjectUpdateInput(body ?? {});

  if (!updates.name && updates.isFavorite === undefined) {
    return NextResponse.json({ message: 'Provide a name or favorite value to update.' }, { status: 400 });
  }

  const routeParams = await params;
  const project = await updateProjectForUser({
    userId: user.id,
    projectId: routeParams.projectId,
    updates,
  });

  if (!project) {
    return NextResponse.json({ message: 'Generation not found.' }, { status: 404 });
  }

  return NextResponse.json({ project });
}

export async function DELETE(_: Request, { params }: ProjectRouteProps) {
  const user = await getAuthenticatedUser();

  if (!user) {
    return NextResponse.json({ message: 'Unauthorized.' }, { status: 401 });
  }

  const routeParams = await params;
  const project = await deleteProjectForUser({
    userId: user.id,
    projectId: routeParams.projectId,
  });

  if (!project) {
    return NextResponse.json({ message: 'Generation not found.' }, { status: 404 });
  }

  const assetPaths = [
    project.sourceImagePath,
    project.maskImagePath,
    project.outputAssetPath,
  ].filter((assetPath): assetPath is string => Boolean(assetPath));
  const referenceCounts = await getProjectAssetReferenceCounts(assetPaths);

  await Promise.all(assetPaths.map(async (assetPath) => {
    if ((referenceCounts.get(assetPath) ?? 0) === 0) {
      await deleteStoredFile(assetPath).catch(() => undefined);
    }
  }));

  return NextResponse.json({ ok: true });
}
