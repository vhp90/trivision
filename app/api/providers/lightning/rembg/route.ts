import path from 'node:path';
import { NextResponse } from 'next/server';
import { getAuthenticatedUser } from '@/lib/auth/session';
import { LightningTrellisClient } from '@/lib/generation/lightning-client';

export async function POST(request: Request) {
  const user = await getAuthenticatedUser();

  if (!user) {
    return NextResponse.json({ message: 'Unauthorized.' }, { status: 401 });
  }

  const formData = await request.formData();
  const sourceFileValue = formData.get('sourceImage');
  const sourceFile = sourceFileValue instanceof File ? sourceFileValue : null;

  if (!sourceFile || sourceFile.size === 0) {
    return NextResponse.json({ message: 'A source image is required.' }, { status: 400 });
  }

  try {
    const client = new LightningTrellisClient();
    const rembgResponse = await client.removeBackground({
      fileName: sourceFile.name || 'source-image.png',
      buffer: Buffer.from(await sourceFile.arrayBuffer()),
      mimeType: sourceFile.type || 'image/png',
    });

    if (!rembgResponse.download_url) {
      return NextResponse.json({ message: 'Lightning background removal did not return an image.' }, { status: 502 });
    }

    const processedAsset = await client.downloadAsset(rembgResponse.download_url);
    const fileName = rembgResponse.filename || `${path.basename(sourceFile.name || 'source-image', path.extname(sourceFile.name || ''))}-rembg.png`;

    return new NextResponse(processedAsset.buffer, {
      status: 200,
      headers: {
        'Content-Type': rembgResponse.content_type || processedAsset.contentType || 'image/png',
        'Content-Disposition': `inline; filename="${fileName}"`,
        'X-Processed-Filename': fileName,
        'Cache-Control': 'no-store',
      },
    });
  } catch (error) {
    return NextResponse.json(
      { message: error instanceof Error ? error.message : 'Background removal failed.' },
      { status: 400 },
    );
  }
}
