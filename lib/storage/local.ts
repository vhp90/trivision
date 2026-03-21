import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';

const storageRoot = path.join(process.cwd(), 'data', 'storage');

function sanitizeSegment(segment: string) {
  return segment.replace(/[^a-zA-Z0-9._-]+/g, '-').replace(/^-+|-+$/g, '') || 'file';
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
  await ensureDirectory(directoryPath);

  const extension = path.extname(input.fileName) || (input.kind === 'outputs' ? '.glb' : '');
  const baseName = sanitizeSegment(path.basename(input.fileName, extension));
  const relativePath = path.join(
    directoryPath,
    `${sanitizeSegment(input.projectId)}-${baseName}-${randomUUID()}${extension}`,
  );

  await writeFile(getAbsoluteStoragePath(relativePath), input.content);

  return relativePath;
}

export async function saveRemoteAsset(input: {
  userId: string;
  projectId: string;
  assetUrl: string;
  outputFormat: string;
}) {
  const response = await fetch(input.assetUrl);

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
  return readFile(getAbsoluteStoragePath(relativePath));
}

export function getContentTypeFromPath(relativePath: string) {
  const extension = path.extname(relativePath).toLowerCase();

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
