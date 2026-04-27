import { describe, expect, it } from 'vitest';
import { getGenerationModel } from '@/lib/generation/registry';
import { normalizeGenerationRequest, normalizeParameterValues } from '@/lib/generation/validation';

describe('generation request normalization', () => {
  it('drops unsupported prompt text instead of rejecting image-only models', () => {
    const normalized = normalizeGenerationRequest(
      {
        modelId: 'microsoft:trellis-2@4b',
        prompt: 'this should not block an image-only run',
        outputFormat: 'GLB',
        parameterValues: {},
      },
      { hasSourceImage: true },
    );

    expect(normalized.payload.prompt).toBe('');
    expect(normalized.payload.outputFormat).toBe('glb');
  });

  it('coerces common form string values and clamps numeric ranges', () => {
    const model = getGenerationModel('lightning:microsoft-trellis-2@4b');
    expect(model).not.toBeNull();

    const values = normalizeParameterValues(model!, {
      seed: '12345',
      decimationTarget: '999999999',
      preprocessImage: 'false',
      remesh: 'false',
      textureSize: '2048',
      pipelineType: 'unsupported-preset',
    });

    expect(values.seed).toBe(12345);
    expect(values.decimationTarget).toBe(2000000);
    expect(values.preprocessImage).toBe(false);
    expect(values.remesh).toBe(false);
    expect(values.textureSize).toBe(2048);
    expect(values.pipelineType).toBe('512');
  });
});
