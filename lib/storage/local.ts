import { mkdir, readFile, unlink, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { del, put } from '@vercel/blob';
import { fetchWithRetry } from '@/lib/http/fetch-with-retry';

const storageRoot = path.join(process.cwd(), 'data', 'storage');

export type StorageMode = 'blob' | 'local';

export function resolveStorageMode(env: Partial<NodeJS.ProcessEnv> = process.env): StorageMode {
  return env.BLOB_READ_WRITE_TOKEN?.trim() ? 'blob' : 'local';
}

function sanitizeSegment(segment: string) {
  return segment.replace(/[^a-zA-Z0-9._-]+/g, '-').replace(/^-+|-+$/g, '') || 'file';
}

function normalizeStoragePath(relativePath: string) {
  return relativePath.replace(/\\/g, '/');
}

function isRemoteAssetPath(filePath: string) {
  return /^https?:\/\//i.test(filePath);
}

function getAbsoluteStoragePath(relativePath: string) {
  return path.join(storageRoot, relativePath);
}

async function ensureDirectory(relativeDirectoryPath: string) {
  const absoluteDirectoryPath = getAbsoluteStoragePath(relativeDirectoryPath);
  await mkdir(absoluteDirectoryPath, { recursive: true });
  return absoluteDirectoryPath;
}

export async function saveUploadedFile(input: {
  userId: string;
  projectId: string;
  kind: 'inputs' | 'outputs';
  fileName: string;
  content: Buffer;
}) {
  const directoryPath = path.join(input.userId, input.kind);
  const extension = path.extname(input.fileName) || (input.kind === 'outputs' ? '.glb' : '');
  const baseName = sanitizeSegment(path.basename(input.fileName, extension));
  const relativePath = path.join(
    directoryPath,
    `${sanitizeSegment(input.projectId)}-${baseName}-${randomUUID()}${extension}`,
  );

  if (resolveStorageMode() === 'blob') {
    const blobPath = normalizeStoragePath(relativePath);
    const blob = await put(blobPath, input.content, {
      access: 'public',
      addRandomSuffix: false,
      contentType: getContentTypeFromPath(blobPath),
    });

    return blob.url;
  }

  await ensureDirectory(directoryPath);
  await writeFile(getAbsoluteStoragePath(relativePath), input.content);

  return relativePath;
}

export async function saveRemoteAsset(input: {
  userId: string;
  projectId: string;
  assetUrl: string;
  outputFormat: string;
}) {
  const response = await fetchWithRetry(input.assetUrl, {
    retries: 2,
    timeoutMs: 45000,
  });

  if (!response.ok) {
    throw new Error('Unable to download generated asset from provider.');
  }

  const content = Buffer.from(await response.arrayBuffer());
  const extension = input.outputFormat.startsWith('.') ? input.outputFormat : `.${input.outputFormat}`;

  return saveUploadedFile({
    userId: input.userId,
    projectId: input.projectId,
    kind: 'outputs',
    fileName: `generated-asset${extension}`,
    content,
  });
}

export async function readStoredFile(relativePath: string) {
  if (isRemoteAssetPath(relativePath)) {
    const response = await fetchWithRetry(relativePath, {
      retries: 2,
      timeoutMs: 30000,
    });

    if (!response.ok) {
      throw new Error('Remote asset could not be read.');
    }

    return Buffer.from(await response.arrayBuffer());
  }

  return readFile(getAbsoluteStoragePath(relativePath));
}

export async function deleteStoredFile(relativePath: string) {
  if (isRemoteAssetPath(relativePath)) {
    await del(relativePath);
    return;
  }

  await unlink(getAbsoluteStoragePath(relativePath)).catch((error: unknown) => {
    if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') {
      return;
    }

    throw error;
  });
}

export function getContentTypeFromPath(relativePath: string) {
  const normalizedPath = isRemoteAssetPath(relativePath)
    ? new URL(relativePath).pathname
    : relativePath;
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
