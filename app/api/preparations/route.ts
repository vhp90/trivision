import { NextResponse } from 'next/server';
import { getAuthenticatedUser } from '@/lib/auth/session';
import {
  getAssetPreparationAssets,
  startAssetPreparation,
} from '@/lib/generation/preparation/service';
import { getFriendlyGenerationError } from '@/lib/generation/errors';

export const runtime = 'nodejs';
export const maxDuration = 60;

export async function POST(request: Request) {
  const user = await getAuthenticatedUser();

  if (!user) {
    return NextResponse.json({ message: 'Unauthorized.' }, { status: 401 });
  }

  const formData = await request.formData();
  const targetModelId = String(formData.get('targetModelId') ?? '').trim();
  const prompt = String(formData.get('prompt') ?? '').trim();
  const textToImage = String(formData.get('textToImage') ?? '') === 'true';
  const removeBackground = String(formData.get('removeBackground') ?? '') === 'true';
  const sourceProjectId = String(formData.get('sourceProjectId') ?? '').trim() || null;
  const sourceFileValue = formData.get('sourceImage');
  const sourceFile = sourceFileValue instanceof File ? sourceFileValue : null;

  if (!targetModelId) {
    return NextResponse.json({ message: 'A target 3D model is required.' }, { status: 400 });
  }

  try {
    const job = await startAssetPreparation({
      userId: user.id,
      targetModelId,
      textToImage,
      removeBackground,
      prompt,
      sourceFile,
      sourceProjectId,
    });

    return NextResponse.json({
      job,
      assets: getAssetPreparationAssets(job),
    }, { status: 202 });
  } catch (error) {
    return NextResponse.json(
      { message: getFriendlyGenerationError(error) },
      { status: 400 },
    );
  }
}
