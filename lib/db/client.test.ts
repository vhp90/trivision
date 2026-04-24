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
    expect(config.isLocalFile).toBe(false);
  });

  it('falls back to the local file database when no remote URL is configured', () => {
    const config = resolveDatabaseConfig({});

    expect(config.url).toContain('trivision.local.db');
    expect(config.url.startsWith('file:')).toBe(true);
    expect(config.isLocalFile).toBe(true);
  });
});
