type FetchWithRetryOptions = RequestInit & {
  retries?: number;
  retryDelayMs?: number;
  timeoutMs?: number;
  retryStatuses?: number[];
};

const defaultRetryStatuses = [408, 429, 500, 502, 503, 504];

function sleep(durationMs: number) {
  return new Promise((resolve) => setTimeout(resolve, durationMs));
}

function isAbortError(error: unknown) {
  return error instanceof Error && error.name === 'AbortError';
}

export async function fetchWithRetry(
  input: string | URL | Request,
  options: FetchWithRetryOptions = {},
) {
  const {
    retries = 2,
    retryDelayMs = 500,
    timeoutMs = 30000,
    retryStatuses = defaultRetryStatuses,
    signal,
    ...requestInit
  } = options;

  let lastError: unknown = null;

  for (let attemptIndex = 0; attemptIndex <= retries; attemptIndex += 1) {
    const timeoutController = new AbortController();
    const timeoutId = setTimeout(() => timeoutController.abort(), timeoutMs);
    const abortFromCaller = () => timeoutController.abort();

    try {
      if (signal) {
        if (signal.aborted) {
          timeoutController.abort();
        } else {
          signal.addEventListener('abort', abortFromCaller, { once: true });
        }
      }

      const response = await fetch(input, {
        ...requestInit,
        signal: timeoutController.signal,
      });

      if (!retryStatuses.includes(response.status) || attemptIndex === retries) {
        return response;
      }

      await response.body?.cancel().catch(() => undefined);
    } catch (error) {
      lastError = error;

      if (isAbortError(error) && signal?.aborted) {
        throw error;
      }

      if (attemptIndex === retries) {
        throw error;
      }
    } finally {
      clearTimeout(timeoutId);
      signal?.removeEventListener('abort', abortFromCaller);
    }

    await sleep(retryDelayMs * (attemptIndex + 1));
  }

  throw lastError instanceof Error ? lastError : new Error('Request failed.');
}
