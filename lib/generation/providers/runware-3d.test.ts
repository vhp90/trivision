import { describe, expect, it } from 'vitest';
import { normalizeRunware3DResult } from '@/lib/generation/providers/runware-3d';
import { getTaskStatus } from '@/lib/generation/runware-client';

describe('runware 3D response helpers', () => {
  it('detects async processing responses without requiring an asset', () => {
    const rawResponse = {
      data: [
        {
          taskType: '3dInference',
          taskUUID: 'task-1',
          status: 'processing',
        },
      ],
    };

    expect(getTaskStatus(rawResponse, 'task-1')).toBe('processing');
  });

  it('normalizes completed 3D outputs from the nested files response shape', () => {
    const rawResponse = {
      data: [
        {
          taskType: '3dInference',
          taskUUID: 'task-2',
          status: 'success',
          outputs: {
            files: [
              {
                uuid: 'asset-1',
                url: 'https://im.runware.ai/asset.glb',
              },
            ],
          },
        },
      ],
    };

    expect(normalizeRunware3DResult(rawResponse, 'task-2')).toMatchObject({
      providerTaskId: 'task-2',
      assetUrl: 'https://im.runware.ai/asset.glb',
      outputFormat: 'glb',
    });
  });
});
