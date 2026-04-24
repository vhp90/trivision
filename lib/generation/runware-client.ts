import { createGenerationTaskId } from '@/lib/generation/helpers';
import { fetchWithRetry } from '@/lib/http/fetch-with-retry';

const RUNWARE_API_URL = process.env.RUNWARE_API_URL ?? 'https://api.runware.ai/v1';

function getRunwareApiKey() {
  const apiKey = process.env.RUNWARE_API_KEY;

  if (!apiKey) {
    throw new Error('RUNWARE_API_KEY is not configured.');
  }

  return apiKey;
}

function bufferToDataUri(buffer: Buffer, mimeType: string) {
  return `data:${mimeType};base64,${buffer.toString('base64')}`;
}

function getResponseData(rawResponse: unknown) {
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

export function findTaskResult(rawResponse: unknown, taskUUID: string) {
  return getResponseData(rawResponse).find((item) => {
    if (!item || typeof item !== 'object') {
      return false;
    }

    return String((item as Record<string, unknown>).taskUUID ?? '') === taskUUID;
  }) as Record<string, unknown> | undefined;
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
      retries: 2,
      timeoutMs: 45000,
    });

    const rawResponse = await response.json().catch(() => null);

    if (!response.ok) {
      throw new Error(getErrorMessage(rawResponse) ?? 'Runware request failed.');
    }

    return rawResponse;
  }

  async uploadImage(buffer: Buffer, mimeType: string) {
    const taskUUID = createGenerationTaskId();
    const rawResponse = await this.request([
      {
        taskType: 'imageUpload',
        taskUUID,
        image: bufferToDataUri(buffer, mimeType),
      },
    ]);

    const result = findTaskResult(rawResponse, taskUUID);

    if (!result) {
      throw new Error('Runware image upload did not return a task result.');
    }

    const imageUUID = result.imageUUID ?? result.imageUuid ?? result.uuid;

    if (typeof imageUUID !== 'string') {
      throw new Error('Runware image upload did not return an image UUID.');
    }

    return imageUUID;
  }
}
