import { describe, expect, it } from 'vitest';
import { resolveStorageMode } from '@/lib/storage/local';

describe('storage mode resolution', () => {
  it('uses Vercel Blob when a Blob token is configured', () => {
    expect(resolveStorageMode({ BLOB_READ_WRITE_TOKEN: 'blob-token' })).toBe('blob');
  });

  it('falls back to local storage without a Blob token', () => {
    expect(resolveStorageMode({})).toBe('local');
  });
});
