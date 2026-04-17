import { generationProviderConfig } from '@/lib/config/app';

type LightningAssetResponse = {
  id?: string;
  filename?: string;
  content_type?: string;
  download_url?: string;
};

type LightningGenerateResponse = {
  id?: string;
  model_id?: string;
  pipeline_type?: string;
  seed?: number;
  elapsed_seconds?: number;
  download_url?: string;
  filename?: string;
  num_vertices?: number;
  num_faces?: number;
};

function getLightningApiUrl() {
  const apiUrl = generationProviderConfig.lightningTrellisApiUrl;

  if (!apiUrl) {
    throw new Error('LIGHTNING_TRELLIS_API_URL is not configured.');
  }

  return apiUrl.replace(/\/+$/, '');
}

async function parseError(response: Response) {
  const raw = await response.text();

  if (!raw) {
    return `Lightning TRELLIS request failed with ${response.status}.`;
  }

  try {
    const payload = JSON.parse(raw) as { detail?: string; message?: string };
    return payload.detail || payload.message || raw;
  } catch {
    return raw;
  }
}

function createImageFormData(fileName: string, buffer: Buffer, mimeType: string) {
  const formData = new FormData();
  formData.append('image', new Blob([new Uint8Array(buffer)], { type: mimeType }), fileName);
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
    this.apiUrl = apiUrl;
  }

  async removeBackground(input: {
    fileName: string;
    buffer: Buffer;
    mimeType: string;
  }) {
    const response = await fetch(`${this.apiUrl}/rembg`, {
      method: 'POST',
      body: createImageFormData(input.fileName, input.buffer, input.mimeType),
    });

    if (!response.ok) {
      throw new Error(await parseError(response));
    }

    return await response.json() as LightningAssetResponse;
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

    const response = await fetch(`${this.apiUrl}/generate`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(await parseError(response));
    }

    return await response.json() as LightningGenerateResponse;
  }

  async downloadAsset(assetUrl: string) {
    const response = await fetch(this.resolveUrl(assetUrl));

    if (!response.ok) {
      throw new Error('Lightning TRELLIS returned an asset URL that could not be downloaded.');
    }

    return {
      buffer: Buffer.from(await response.arrayBuffer()),
      contentType: response.headers.get('content-type') || 'application/octet-stream',
    };
  }

  resolveUrl(assetUrl: string) {
    const resolvedUrl = new URL(assetUrl, this.apiUrl);
    const apiUrl = new URL(this.apiUrl);

    if (
      resolvedUrl.hostname === 'localhost'
      || resolvedUrl.hostname === '127.0.0.1'
      || resolvedUrl.hostname === '0.0.0.0'
    ) {
      resolvedUrl.protocol = apiUrl.protocol;
      resolvedUrl.host = apiUrl.host;
    }

    return resolvedUrl.toString();
  }
}
