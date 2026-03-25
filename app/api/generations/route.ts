import { NextResponse } from 'next/server';
import { getAuthenticatedUser } from '@/lib/auth/session';
import { startGeneration } from '@/lib/generation/service';
import type { GenerationParameterValueMap } from '@/lib/generation/types';

export async function POST(request: Request) {
  const user = await getAuthenticatedUser();

  if (!user) {
    return NextResponse.json({ message: 'Unauthorized.' }, { status: 401 });
  }

  const formData = await request.formData();
  const modelId = String(formData.get('modelId') ?? '').trim();
  const prompt = String(formData.get('prompt') ?? '').trim();
  const outputFormat = String(formData.get('outputFormat') ?? 'glb').trim();
  const sourceProjectId = String(formData.get('sourceProjectId') ?? '').trim() || null;
  const rawParameters = String(formData.get('parameters') ?? '{}');
  const sourceFileValue = formData.get('sourceImage');
  const maskFileValue = formData.get('maskImage');
  const sourceFile = sourceFileValue instanceof File ? sourceFileValue : null;
  const maskFile = maskFileValue instanceof File ? maskFileValue : null;

  if (!modelId) {
    return NextResponse.json({ message: 'A generation model is required.' }, { status: 400 });
  }

  let parameterValues: GenerationParameterValueMap = {};

  try {
    const parsedParameters = JSON.parse(rawParameters);
    parameterValues = parsedParameters && typeof parsedParameters === 'object'
      ? parsedParameters as GenerationParameterValueMap
      : {};
  } catch {
    return NextResponse.json({ message: 'Generation parameters are invalid.' }, { status: 400 });
  }

  try {
    const result = await startGeneration({
      userId: user.id,
      modelId,
      prompt,
      outputFormat,
      parameterValues,
      sourceFile,
      maskFile,
      sourceProjectId,
    });

    return NextResponse.json(result, { status: 202 });
  } catch (error) {
    return NextResponse.json(
      { message: error instanceof Error ? error.message : 'Unable to start generation.' },
      { status: 400 },
    );
  }
}
