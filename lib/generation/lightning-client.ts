import { generationProviderConfig } from '@/lib/config/app';
import { fetchWithRetry } from '@/lib/http/fetch-with-retry';

export type LightningGenerateResponse = {
  job_id?: string;
  queue_position?: number;
  status?: string;
};

export type LightningJobStatusResponse = {
  job_id?: string;
  status?: string;
  elapsed_time?: number | null;
  stage_times?: Record<string, number> | null;
};

function normalizeLightningApiUrl(apiUrl: string) {
  const parsedUrl = new URL(apiUrl);

  if (parsedUrl.pathname === '/' || parsedUrl.pathname === '') {
    parsedUrl.pathname = '/api';
  }

  return parsedUrl.toString().replace(/\/+$/, '');
}

function getLightningApiUrl() {
  const apiUrl = generationProviderConfig.lightningTrellisApiUrl;
  if (!apiUrl) {
    throw new Error('LIGHTNING_TRELLIS_API_URL is not configured.');
  }

  return normalizeLightningApiUrl(apiUrl);
}

async function parseError(response: Response) {
  const raw = await response.text();

  if (!raw) {
    return `Lightning TRELLIS request failed with ${response.status}.`;
  }

  try {
    const payload = JSON.parse(raw) as { detail?: string; message?: string; error?: string };
    return payload.error || payload.detail || payload.message || raw;
  } catch {
    return raw;
  }
}

function createImageFormData(fileName: string, buffer: Buffer, mimeType: string) {
  const formData = new FormData();
  formData.append('file', new Blob([new Uint8Array(buffer)], { type: mimeType }), fileName);
  return formData;
}

function appendOptionalFormField(formData: FormData, key: string, value: string | number | boolean | null | undefined) {
  if (value === null || value === undefined || value === '') {
    return;
  }

  formData.append(key, String(value));
}

export class LightningTrellisClient {
  private readonly apiUrl: string;

  constructor(apiUrl = getLightningApiUrl()) {
    this.apiUrl = normalizeLightningApiUrl(apiUrl);
  }

  get baseUrl() {
    return this.apiUrl;
  }

  async generate(input: {
    fileName: string;
    buffer: Buffer;
    mimeType: string;
    parameters: Record<string, string | number | boolean | null | undefined>;
  }) {
    const formData = createImageFormData(input.fileName, input.buffer, input.mimeType);

    for (const [key, value] of Object.entries(input.parameters)) {
      appendOptionalFormField(formData, key, value);
    }

    const response = await fetchWithRetry(`${this.apiUrl}/generate`, {
      method: 'POST',
      body: formData,
      retries: 1,
      timeoutMs: 45000,
    });

    if (!response.ok) {
      throw new Error(await parseError(response));
    }

    return await response.json() as LightningGenerateResponse;
  }

  async getJobStatus(jobId: string) {
    const response = await fetchWithRetry(`${this.apiUrl}/job/${encodeURIComponent(jobId)}/status`, {
      retries: 2,
      timeoutMs: 20000,
    });

    if (!response.ok) {
      throw new Error(await parseError(response));
    }

    return await response.json() as LightningJobStatusResponse;
  }

  getResultUrl(jobId: string) {
    return `${this.apiUrl}/job/${encodeURIComponent(jobId)}/result`;
  }
}
