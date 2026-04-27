import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { del, put } from '@vercel/blob';
import { fetchWithRetry } from '@/lib/http/fetch-with-retry';

export function resolveBlobStorageConfig(env: Partial<NodeJS.ProcessEnv> = process.env) {
  const token = env.BLOB_READ_WRITE_TOKEN?.trim();

  if (!token) {
    throw new Error('BLOB_READ_WRITE_TOKEN is required. Configure Vercel Blob before uploading assets.');
  }

  return { token };
}

function sanitizeSegment(segment: string) {
  return segment.replace(/[^a-zA-Z0-9._-]+/g, '-').replace(/^-+|-+$/g, '') || 'file';
}

function normalizeBlobPath(blobPath: string) {
  return blobPath.replace(/\\/g, '/');
}

function isRemoteAssetPath(filePath: string) {
  return /^https?:\/\//i.test(filePath);
}

function assertRemoteAssetPath(filePath: string) {
  if (!isRemoteAssetPath(filePath)) {
    throw new Error('Stored asset is not a Vercel Blob URL.');
  }
}

export async function saveUploadedFile(input: {
  userId: string;
  projectId: string;
  kind: 'inputs' | 'outputs';
  fileName: string;
  content: Buffer;
}) {
  resolveBlobStorageConfig();

  const extension = path.extname(input.fileName) || (input.kind === 'outputs' ? '.glb' : '');
  const baseName = sanitizeSegment(path.basename(input.fileName, extension));
  const blobPath = normalizeBlobPath(path.join(
    sanitizeSegment(input.userId),
    input.kind,
    `${sanitizeSegment(input.projectId)}-${baseName}-${randomUUID()}${extension}`,
  ));
  const blob = await put(blobPath, input.content, {
    access: 'public',
    addRandomSuffix: false,
    contentType: getContentTypeFromPath(blobPath),
  });

  return blob.url;
}

export async function saveRemoteAsset(input: {
  userId: string;
  projectId: string;
  assetUrl: string;
  outputFormat: string;
}) {
  return saveRemoteFile({
    userId: input.userId,
    projectId: input.projectId,
    kind: 'outputs',
    assetUrl: input.assetUrl,
    fileName: `generated-asset.${input.outputFormat.replace(/^\./, '')}`,
  });
}

export async function saveRemoteFile(input: {
  userId: string;
  projectId: string;
  kind: 'inputs' | 'outputs';
  assetUrl: string;
  fileName: string;
}) {
  const response = await fetchWithRetry(input.assetUrl, {
    retries: 1,
    timeoutMs: 25000,
  });

  if (!response.ok) {
    throw new Error('Unable to download generated asset from provider.');
  }

  const content = Buffer.from(await response.arrayBuffer());

  return saveUploadedFile({
    userId: input.userId,
    projectId: input.projectId,
    kind: input.kind,
    fileName: input.fileName,
    content,
  });
}

export async function readStoredFile(assetUrl: string) {
  assertRemoteAssetPath(assetUrl);

  const response = await fetchWithRetry(assetUrl, {
    retries: 2,
    timeoutMs: 30000,
  });

  if (!response.ok) {
    throw new Error('Blob asset could not be read.');
  }

  return Buffer.from(await response.arrayBuffer());
}

export async function deleteStoredFile(assetUrl: string) {
  assertRemoteAssetPath(assetUrl);
  await del(assetUrl);
}

export function getContentTypeFromPath(filePath: string) {
  const normalizedPath = isRemoteAssetPath(filePath)
    ? new URL(filePath).pathname
    : filePath;
  const extension = path.extname(normalizedPath).toLowerCase();

  switch (extension) {
    case '.png':
      return 'image/png';
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg';
    case '.webp':
      return 'image/webp';
    case '.glb':
      return 'model/gltf-binary';
    case '.gltf':
      return 'model/gltf+json';
    default:
      return 'application/octet-stream';
  }
}
