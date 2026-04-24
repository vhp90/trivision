import { describe, expect, it } from 'vitest';
import { getFriendlyGenerationError } from '@/lib/generation/errors';

describe('friendly generation errors', () => {
  it('turns missing provider configuration into a demo-safe message', () => {
    expect(getFriendlyGenerationError(new Error('RUNWARE_API_KEY is not configured.')))
      .toBe('Generation provider is not configured. Check the demo environment settings and try again.');
  });

  it('turns provider download failures into a helpful retry message', () => {
    expect(getFriendlyGenerationError(new Error('Runware returned a 3D response without a downloadable asset URL.')))
      .toBe('The provider finished without a downloadable asset. Retry the generation or adjust the input image.');
  });

  it('turns provider parameter failures into an action-oriented message', () => {
    expect(getFriendlyGenerationError(new Error("Missing required parameter 'positivePrompt'.")))
      .toBe('The provider could not use those settings. Adjust the prompt or parameters, then retry.');
  });
});
