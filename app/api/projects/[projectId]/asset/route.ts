import path from 'node:path';
import { NextResponse } from 'next/server';
import { getAuthenticatedUser } from '@/lib/auth/session';
import { getProjectFilePathForUser } from '@/lib/db/repository';
import { getContentTypeFromPath, readStoredFile } from '@/lib/storage/local';

type ProjectAssetRouteProps = {
  params: Promise<{
    projectId: string;
  }>;
};

export async function GET(request: Request, { params }: ProjectAssetRouteProps) {
  const user = await getAuthenticatedUser();

  if (!user) {
    return NextResponse.json({ message: 'Unauthorized.' }, { status: 401 });
  }

  const routeParams = await params;
  const url = new URL(request.url);
  const requestedKind = url.searchParams.get('kind');
  const shouldDownload = url.searchParams.get('download') === '1';
  const requestedFilename = url.searchParams.get('filename');
  const kind = requestedKind === 'source' ? 'source' : 'output';
  const relativePath = await getProjectFilePathForUser({
    userId: user.id,
    projectId: routeParams.projectId,
    kind,
  });

  if (!relativePath) {
    return NextResponse.json({ message: 'Asset not found.' }, { status: 404 });
  }

  try {
    const fileBuffer = await readStoredFile(relativePath);
    return new NextResponse(fileBuffer, {
      status: 200,
      headers: {
        'Content-Type': getContentTypeFromPath(relativePath),
        'Cache-Control': 'private, max-age=60',
        ...(shouldDownload
          ? {
            'Content-Disposition': `attachment; filename="${requestedFilename || path.basename(relativePath)}"`,
          }
          : {}),
      },
    });
  } catch {
    return NextResponse.json({ message: 'Asset could not be read.' }, { status: 404 });
  }
}
