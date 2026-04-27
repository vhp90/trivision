import { createClient } from '@libsql/client';
import {
  createEmptyWorkspace,
  defaultSettingSections,
} from '@/lib/db/seed';
import type {
  SettingSection,
  UserProfile,
  WorkspaceSummary,
} from '@/lib/db/types';

const SCHEMA_VERSION = '4';

let initializationPromise: Promise<void> | null = null;
let client: ReturnType<typeof createClient> | null = null;

export function resolveDatabaseConfig(env: Partial<NodeJS.ProcessEnv> = process.env) {
  const remoteUrl = env.DATABASE_URL?.trim() || env.TURSO_DATABASE_URL?.trim();
  const authToken = env.DATABASE_AUTH_TOKEN?.trim() || env.TURSO_AUTH_TOKEN?.trim();

  if (!remoteUrl) {
    throw new Error('DATABASE_URL is required. Configure the Turso/libSQL URL in Vercel and .env.local.');
  }

  if (remoteUrl.startsWith('file:')) {
    throw new Error('Local file databases are no longer supported. Use the Turso/libSQL DATABASE_URL.');
  }

  if (remoteUrl.startsWith('libsql://') && !authToken) {
    throw new Error('DATABASE_AUTH_TOKEN is required for Turso/libSQL databases.');
  }

  return {
    url: remoteUrl,
    authToken: authToken || undefined,
  };
}

function getClientInstance() {
  if (!client) {
    const databaseConfig = resolveDatabaseConfig();
    client = createClient(databaseConfig.authToken
      ? { url: databaseConfig.url, authToken: databaseConfig.authToken }
      : { url: databaseConfig.url });
  }

  return client;
}

async function insertUser(clientInstance: ReturnType<typeof createClient>, user: UserProfile, passwordHash: string) {
  await clientInstance.execute({
    sql: `
      INSERT INTO users (
        id, email, password_hash, full_name, initials, role_label, region,
        latency_label, session_label, engine_version, unread_notifications
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `,
    args: [
      user.id,
      user.email,
      passwordHash,
      user.fullName,
      user.initials,
      user.roleLabel,
      user.region,
      user.latencyLabel,
      user.sessionLabel,
      user.engineVersion,
      user.unreadNotifications,
    ],
  });
}

async function insertWorkspace(clientInstance: ReturnType<typeof createClient>, workspace: WorkspaceSummary) {
  await clientInstance.execute({
    sql: `
      INSERT INTO workspaces (
        id, user_id, name, code, description, status, project_count, favorite_count,
        updated_label, primary_focus, secondary_focus, is_primary
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `,
    args: [
      workspace.id,
      workspace.userId,
      workspace.name,
      workspace.code,
      workspace.description,
      workspace.status,
      workspace.projectCount,
      workspace.favoriteCount,
      workspace.updatedLabel,
      workspace.primaryFocus,
      workspace.secondaryFocus,
      workspace.isPrimary ? 1 : 0,
    ],
  });
}

async function insertSettingSections(
  clientInstance: ReturnType<typeof createClient>,
  userId: string,
  sections: SettingSection[],
) {
  for (const section of sections) {
    for (const item of section.items) {
      await clientInstance.execute({
        sql: `
          INSERT OR IGNORE INTO settings (
            id, user_id, section_id, section_title, section_description, label, value, description
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        `,
        args: [
          `${userId}-${item.id}`,
          userId,
          section.id,
          section.title,
          section.description,
          item.label,
          item.value,
          item.description,
        ],
      });
    }
  }
}

async function bootstrapUserData(clientInstance: ReturnType<typeof createClient>, user: UserProfile) {
  await insertWorkspace(clientInstance, createEmptyWorkspace(user));
  await insertSettingSections(clientInstance, user.id, defaultSettingSections);
}

async function initializeDatabase() {
  const clientInstance = getClientInstance();

  await clientInstance.execute(`
    CREATE TABLE IF NOT EXISTS app_meta (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    )
  `);

  await clientInstance.batch(
    [
      {
        sql: `
          CREATE TABLE IF NOT EXISTS app_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
          )
        `,
      },
      {
        sql: `
          CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            initials TEXT NOT NULL,
            role_label TEXT NOT NULL,
            region TEXT NOT NULL,
            latency_label TEXT NOT NULL,
            session_label TEXT NOT NULL,
            engine_version TEXT NOT NULL,
            unread_notifications INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
          )
        `,
      },
      {
        sql: `
          CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
          )
        `,
      },
      {
        sql: `
          CREATE TABLE IF NOT EXISTS workspaces (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            code TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL,
            project_count INTEGER NOT NULL,
            favorite_count INTEGER NOT NULL,
            updated_label TEXT NOT NULL,
            primary_focus TEXT NOT NULL,
            secondary_focus TEXT NOT NULL,
            is_primary INTEGER NOT NULL DEFAULT 0
          )
        `,
      },
      {
        sql: `
          CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            workspace_name TEXT NOT NULL,
            name TEXT NOT NULL,
            format TEXT,
            updated_label TEXT NOT NULL,
            tris_label TEXT NOT NULL,
            visual TEXT NOT NULL,
            prompt TEXT NOT NULL,
            seed TEXT NOT NULL,
            resolution TEXT NOT NULL,
            creativity INTEGER NOT NULL,
            detail_level TEXT NOT NULL,
            tri_count TEXT NOT NULL,
            vert_count TEXT NOT NULL,
            fps TEXT NOT NULL,
            auto_save_label TEXT NOT NULL,
            is_favorite INTEGER NOT NULL DEFAULT 0,
            is_recent INTEGER NOT NULL DEFAULT 1,
            sort_order INTEGER NOT NULL DEFAULT 999,
            generation_status TEXT NOT NULL DEFAULT 'succeeded',
            provider_id TEXT,
            model_id TEXT,
            generation_job_id TEXT,
            parameter_values_json TEXT NOT NULL DEFAULT '{}',
            source_image_path TEXT,
            mask_image_path TEXT,
            output_asset_path TEXT,
            output_format TEXT,
            error_message TEXT,
            submitted_at TEXT,
            completed_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
          )
        `,
      },
      {
        sql: `
          CREATE TABLE IF NOT EXISTS generation_jobs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            provider_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            status TEXT NOT NULL,
            provider_task_id TEXT,
            request_payload_json TEXT NOT NULL,
            response_payload_json TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            started_at TEXT,
            completed_at TEXT
          )
        `,
      },
      {
        sql: `
          CREATE TABLE IF NOT EXISTS settings (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            section_id TEXT NOT NULL,
            section_title TEXT NOT NULL,
            section_description TEXT NOT NULL,
            label TEXT NOT NULL,
            value TEXT NOT NULL,
            description TEXT NOT NULL
          )
        `,
      },
      {
        sql: `
          INSERT OR REPLACE INTO app_meta (key, value) VALUES ('schema_version', '${SCHEMA_VERSION}')
        `,
      },
    ],
    'write',
  );

}

export async function getDatabaseClient() {
  if (!initializationPromise) {
    initializationPromise = initializeDatabase();
  }

  await initializationPromise;
  return getClientInstance();
}

export async function provisionNewUser(user: UserProfile, passwordHash: string) {
  const clientInstance = await getDatabaseClient();
  await insertUser(clientInstance, user, passwordHash);
  await bootstrapUserData(clientInstance, user);
}
