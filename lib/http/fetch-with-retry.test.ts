import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchWithRetry } from '@/lib/http/fetch-with-retry';

describe('fetchWithRetry', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('retries transient failed responses before returning success', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response('try again', { status: 503 }))
      .mockResolvedValueOnce(new Response('ok', { status: 200 }));

    const response = await fetchWithRetry('https://example.com/model.glb', {
      retries: 1,
      retryDelayMs: 1,
      timeoutMs: 1000,
    });

    expect(await response.text()).toBe('ok');
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('does not retry non-transient client errors', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response('bad request', { status: 400 }));

    const response = await fetchWithRetry('https://example.com/model.glb', {
      retries: 3,
      retryDelayMs: 1,
      timeoutMs: 1000,
    });

    expect(response.status).toBe(400);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
