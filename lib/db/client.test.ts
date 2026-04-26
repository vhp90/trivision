import { describe, expect, it } from 'vitest';
import { resolveDatabaseConfig } from '@/lib/db/client';

describe('database configuration', () => {
  it('uses remote libSQL configuration when DATABASE_URL is provided', () => {
    const config = resolveDatabaseConfig({
      DATABASE_URL: 'libsql://trivision-example.turso.io',
      DATABASE_AUTH_TOKEN: 'secret-token',
    });

    expect(config.url).toBe('libsql://trivision-example.turso.io');
    expect(config.authToken).toBe('secret-token');
  });

  it('requires a hosted database URL', () => {
    expect(() => resolveDatabaseConfig({})).toThrow('DATABASE_URL is required');
  });

  it('rejects local file databases', () => {
    expect(() => resolveDatabaseConfig({
      DATABASE_URL: 'file:./data/trivision.local.db',
    })).toThrow('Local file databases are no longer supported');
  });

  it('requires a Turso auth token for libSQL URLs', () => {
    expect(() => resolveDatabaseConfig({
      DATABASE_URL: 'libsql://trivision-example.turso.io',
    })).toThrow('DATABASE_AUTH_TOKEN is required');
  });
});
