import { fetchWithRetry } from '@/lib/http/fetch-with-retry';

const RUNWARE_API_URL = process.env.RUNWARE_API_URL ?? 'https://api.runware.ai/v1';

type RunwareApiErrorOptions = {
  code?: string;
  taskUUID?: string;
  status?: number;
  rawResponse?: unknown;
};

export class RunwareApiError extends Error {
  readonly code?: string;
  readonly taskUUID?: string;
  readonly status?: number;
  readonly rawResponse?: unknown;

  constructor(message: string, options: RunwareApiErrorOptions = {}) {
    super(message);
    this.name = 'RunwareApiError';
    this.code = options.code;
    this.taskUUID = options.taskUUID;
    this.status = options.status;
    this.rawResponse = options.rawResponse;
  }
}

function getRunwareApiKey() {
  const apiKey = process.env.RUNWARE_API_KEY;

  if (!apiKey) {
    throw new Error('RUNWARE_API_KEY is not configured.');
  }

  return apiKey;
}

export function getResponseData(rawResponse: unknown) {
  if (!rawResponse || typeof rawResponse !== 'object') {
    return [];
  }

  const responseRecord = rawResponse as Record<string, unknown>;
  const data = responseRecord.data;

  if (Array.isArray(data)) {
    return data;
  }

  if (data && typeof data === 'object') {
    return [data];
  }

  return [];
}

function getResponseErrors(rawResponse: unknown) {
  if (!rawResponse || typeof rawResponse !== 'object') {
    return [];
  }

  const responseRecord = rawResponse as Record<string, unknown>;
  return Array.isArray(responseRecord.errors) ? responseRecord.errors : [];
}

function getErrorMessage(rawResponse: unknown) {
  if (!rawResponse || typeof rawResponse !== 'object') {
    return null;
  }

  const responseRecord = rawResponse as Record<string, unknown>;

  if (typeof responseRecord.message === 'string') {
    return responseRecord.message;
  }

  if (Array.isArray(responseRecord.errors) && typeof responseRecord.errors[0] === 'string') {
    return responseRecord.errors[0];
  }

  if (Array.isArray(responseRecord.errors) && responseRecord.errors[0] && typeof responseRecord.errors[0] === 'object') {
    const firstError = responseRecord.errors[0] as Record<string, unknown>;

    if (typeof firstError.message === 'string') {
      return firstError.message;
    }
  }

  return null;
}

function getRunwareError(rawResponse: unknown) {
  const errors = getResponseErrors(rawResponse);
  const firstError = errors[0];

  if (!firstError) {
    return null;
  }

  if (typeof firstError === 'string') {
    return {
      message: firstError,
      code: undefined,
      taskUUID: undefined,
    };
  }

  if (typeof firstError === 'object') {
    const errorRecord = firstError as Record<string, unknown>;
    return {
      message: typeof errorRecord.message === 'string'
        ? errorRecord.message
        : 'Runware request failed.',
      code: typeof errorRecord.code === 'string' ? errorRecord.code : undefined,
      taskUUID: typeof errorRecord.taskUUID === 'string' ? errorRecord.taskUUID : undefined,
    };
  }

  return null;
}

export function findTaskResult(rawResponse: unknown, taskUUID: string) {
  return getResponseData(rawResponse).find((item) => {
    if (!item || typeof item !== 'object') {
      return false;
    }

    return String((item as Record<string, unknown>).taskUUID ?? '') === taskUUID;
  }) as Record<string, unknown> | undefined;
}

export function getTaskStatus(rawResponse: unknown, taskUUID: string) {
  const taskResult = findTaskResult(rawResponse, taskUUID);
  const status = taskResult?.status;
  return typeof status === 'string' ? status.toLowerCase() : null;
}

export class RunwareClient {
  private readonly apiKey: string;

  constructor(apiKey = getRunwareApiKey()) {
    this.apiKey = apiKey;
  }

  async request(tasks: Array<Record<string, unknown>>) {
    const response = await fetchWithRetry(RUNWARE_API_URL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(tasks),
      retries: 1,
      timeoutMs: 20000,
    });

    const rawResponse = await response.json().catch(() => null);
    const apiError = getRunwareError(rawResponse);

    if (!response.ok || apiError) {
      throw new RunwareApiError(
        apiError?.message ?? getErrorMessage(rawResponse) ?? 'Runware request failed.',
        {
          code: apiError?.code,
          taskUUID: apiError?.taskUUID,
          status: response.status,
          rawResponse,
        },
      );
    }

    return rawResponse;
  }

  async getResponse(taskUUID: string) {
    return this.request([
      {
        taskType: 'getResponse',
        taskUUID,
      },
    ]);
  }
}
