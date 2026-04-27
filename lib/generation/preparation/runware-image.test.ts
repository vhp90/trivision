import { describe, expect, it } from 'vitest';
import {
  buildFluxKleinRequest,
  buildRmbgRequest,
  normalizeRunwareImageResult,
} from '@/lib/generation/preparation/runware-image';

describe('Runware image preparation requests', () => {
  it('builds a cost-conscious FLUX.2 klein text-to-image request', () => {
    expect(buildFluxKleinRequest({
      taskUUID: 'task-flux',
      prompt: 'a clean product render of a sci-fi helmet',
    })).toEqual({
      taskType: 'imageInference',
      taskUUID: 'task-flux',
      model: 'runware:400@6',
      positivePrompt: 'a clean product render of a sci-fi helmet',
      width: 1024,
      height: 1024,
      outputFormat: 'PNG',
      numberResults: 1,
    });
  });

  it('builds an RMBG request that preserves transparent output', () => {
    expect(buildRmbgRequest({
      taskUUID: 'task-rmbg',
      image: 'https://example.com/source.png',
    })).toEqual({
      taskType: 'removeBackground',
      taskUUID: 'task-rmbg',
      model: 'bria:2@1',
      inputs: {
        image: 'https://example.com/source.png',
      },
      outputFormat: 'PNG',
    });
  });

  it('normalizes image result URLs from common Runware response shapes', () => {
    const rawResponse = {
      data: [
        {
          taskUUID: 'task-image',
          status: 'success',
          imageURL: 'https://im.runware.ai/generated.png',
        },
      ],
    };

    expect(normalizeRunwareImageResult(rawResponse, 'task-image')).toMatchObject({
      providerTaskId: 'task-image',
      imageUrl: 'https://im.runware.ai/generated.png',
      outputFormat: 'png',
    });
  });
});
