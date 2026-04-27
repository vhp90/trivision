import { describe, expect, it } from 'vitest';
import { buildLightningParameters, normalizeLightningJobResult, normalizeLightningSubmitResult } from '@/lib/generation/providers/trellis-lightning';
import { getGenerationModel, getModelParameterDefaults } from '@/lib/generation/registry';

describe('Lightning TRELLIS provider', () => {
  it('uses the documented 512 pipeline defaults and only documented API fields', () => {
    const model = getGenerationModel('lightning:microsoft-trellis-2@4b');

    expect(model).toBeTruthy();

    const defaults = getModelParameterDefaults(model!);
    const parameters = buildLightningParameters({ parameterValues: defaults });

    expect(parameters).toEqual({
      seed: 42,
      pipeline_type: '512',
      preprocess_image: true,
      decimation_target: 1000000,
      texture_size: 4096,
      remesh: true,
      simplify_limit: 16777216,
    });
  });

  it('normalizes queued submit responses from /api/generate', () => {
    expect(normalizeLightningSubmitResult({
      job_id: 'job-123',
      queue_position: 1,
      status: 'queued',
    })).toEqual({
      status: 'running',
      providerTaskId: 'job-123',
      rawResponse: {
        job_id: 'job-123',
        queue_position: 1,
        status: 'queued',
      },
    });
  });

  it('normalizes completed status responses with a result endpoint asset URL', () => {
    expect(normalizeLightningJobResult({
      job_id: 'job-123',
      status: 'complete',
      elapsed_time: 130,
    }, 'https://example.com/api')).toMatchObject({
      providerTaskId: 'job-123',
      assetUrl: 'https://example.com/api/job/job-123/result',
      outputFormat: 'glb',
    });
  });
});
