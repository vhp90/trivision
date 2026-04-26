import { describe, expect, it } from 'vitest';
import { resolveBlobStorageConfig } from '@/lib/storage/blob';

describe('blob storage configuration', () => {
  it('uses Vercel Blob when a Blob token is configured', () => {
    expect(resolveBlobStorageConfig({ BLOB_READ_WRITE_TOKEN: 'blob-token' })).toEqual({
      token: 'blob-token',
    });
  });

  it('requires a Blob token', () => {
    expect(() => resolveBlobStorageConfig({})).toThrow('BLOB_READ_WRITE_TOKEN is required');
  });
});
