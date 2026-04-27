import { describe, expect, it } from 'vitest';
import { generationModels, getModelParameterDefaults } from '@/lib/generation/registry';

describe('generation model registry', () => {
  it('keeps every enabled model self-describing for dynamic controls', () => {
    for (const model of generationModels.filter((entry) => entry.availability === 'enabled')) {
      expect(model.id).toBeTruthy();
      expect(model.providerId).toBeTruthy();
      expect(model.defaultOutputFormat).toBeTruthy();
      expect(model.capabilities.outputFormats).toContain(model.defaultOutputFormat);
      expect(Array.isArray(model.parameters)).toBe(true);
    }
  });

  it('builds defaults directly from each model parameter schema', () => {
    for (const model of generationModels) {
      const defaults = getModelParameterDefaults(model);

      for (const parameter of model.parameters) {
        expect(defaults[parameter.key]).toBe(parameter.defaultValue);
      }
    }
  });

  it('keeps Runware model requests on provider defaults unless explicit controls are added', () => {
    const runwareModels = generationModels.filter((model) => model.providerId.startsWith('runware:'));

    for (const model of runwareModels) {
      expect(model.parameters).toEqual([]);
      expect(getModelParameterDefaults(model)).toEqual({});
    }
  });
});
