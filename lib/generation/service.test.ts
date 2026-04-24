import { describe, expect, it } from 'vitest';
import { resolveGenerationProcessingMode } from '@/lib/generation/service';

describe('generation processing mode', () => {
  it('uses synchronous processing on Vercel by default', () => {
    expect(resolveGenerationProcessingMode({ VERCEL: '1' })).toBe('sync');
  });

  it('keeps local development background processing by default', () => {
    expect(resolveGenerationProcessingMode({})).toBe('background');
  });

  it('allows an explicit mode override', () => {
    expect(resolveGenerationProcessingMode({
      VERCEL: '1',
      GENERATION_PROCESSING_MODE: 'background',
    })).toBe('background');
  });
});
