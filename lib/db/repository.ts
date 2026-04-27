import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { hashPassword, verifyPassword } from '@/lib/auth/password';
import { newUserProfileDefaults, projectRecordDefaults } from '@/lib/config/app';
import { defaultSettingSections } from '@/lib/db/seed';
import { getGenerationModel, getModelParameterDefaults } from '@/lib/generation/registry';
import type { GenerationParameterValueMap, GenerationRequestPayload } from '@/lib/generation/types';
import { getDatabaseClient, provisionNewUser } from '@/lib/db/client';
import type { ProjectUpdateInput } from '@/lib/db/project-actions';
import { normalizeProjectUpdateInput } from '@/lib/db/project-actions';
import type {
  GenerationJobSummary,
  LoginPayload,
  ProjectRecord,
  SettingItem,
  SettingSection,
  ShellSummary,
  SignupPayload,
  UserProfile,
  VisualKind,
  WorkspaceSummary,
} from '@/lib/db/types';

function safeParseParameterValues(rawValue: unknown): GenerationParameterValueMap {
  if (typeof rawValue !== 'string') {
    return {};
  }

  try {
    const parsed = JSON.parse(rawValue);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? parsed as GenerationParameterValueMap
      : {};
  } catch {
    return {};
  }
}

function createInitials(fullName: string) {
  return fullName
    .split(/\s+/)
    .slice(0, 2)
    .map((segment) => segment[0]?.toUpperCase() ?? '')
    .join('')
    || 'NV';
}

function mapUserProfile(row: Record<string, unknown>): UserProfile {
  return {
    id: String(row.id),
    fullName: String(row.full_name),
    email: String(row.email),
    initials: String(row.initials),
    roleLabel: String(row.role_label),
    region: String(row.region),
    latencyLabel: String(row.latency_label),
    sessionLabel: String(row.session_label),
    engineVersion: String(row.engine_version),
    unreadNotifications: Number(row.unread_notifications),
  };
}

function mapWorkspace(row: Record<string, unknown>): WorkspaceSummary {
  return {
    id: String(row.id),
    userId: String(row.user_id),
    name: String(row.name),
    code: String(row.code),
    description: String(row.description),
    status: String(row.status),
    projectCount: Number(row.project_count),
    favoriteCount: Number(row.favorite_count),
    updatedLabel: String(row.updated_label),
    primaryFocus: String(row.primary_focus),
    secondaryFocus: String(row.secondary_focus),
    isPrimary: Number(row.is_primary) === 1,
  };
}

function mapProject(row: Record<string, unknown>): ProjectRecord {
  return {
    id: String(row.id),
    userId: String(row.user_id),
    workspaceId: String(row.workspace_id),
    workspaceName: String(row.workspace_name),
    name: String(row.name),
    format: row.format ? String(row.format) : null,
    updatedLabel: String(row.updated_label),
    trisLabel: String(row.tris_label),
    visual: String(row.visual) as VisualKind,
    prompt: String(row.prompt),
    seed: String(row.seed),
    resolution: String(row.resolution),
    creativity: Number(row.creativity),
    detailLevel: String(row.detail_level),
    triCount: String(row.tri_count),
    vertCount: String(row.vert_count),
    fps: String(row.fps),
    autoSaveLabel: String(row.auto_save_label),
    isFavorite: Number(row.is_favorite) === 1,
    isRecent: Number(row.is_recent) === 1,
    status: String(row.generation_status) as ProjectRecord['status'],
    providerId: row.provider_id ? String(row.provider_id) : null,
    modelId: row.model_id ? String(row.model_id) : null,
    generationJobId: row.generation_job_id ? String(row.generation_job_id) : null,
    parameterValues: safeParseParameterValues(row.parameter_values_json),
    sourceImagePath: row.source_image_path ? String(row.source_image_path) : null,
    maskImagePath: row.mask_image_path ? String(row.mask_image_path) : null,
    outputAssetPath: row.output_asset_path ? String(row.output_asset_path) : null,
    outputFormat: row.output_format ? String(row.output_format) : null,
    errorMessage: row.error_message ? String(row.error_message) : null,
    submittedAt: row.submitted_at ? String(row.submitted_at) : null,
    completedAt: row.completed_at ? String(row.completed_at) : null,
  };
}

function mapGenerationJob(row: Record<string, unknown>): GenerationJobSummary {
  return {
    id: String(row.id),
    projectId: String(row.project_id),
    userId: String(row.user_id),
    providerId: String(row.provider_id),
    modelId: String(row.model_id),
    status: String(row.status) as GenerationJobSummary['status'],
    providerTaskId: row.provider_task_id ? String(row.provider_task_id) : null,
    requestPayloadJson: String(row.request_payload_json),
    responsePayloadJson: row.response_payload_json ? String(row.response_payload_json) : null,
    attemptCount: Number(row.attempt_count),
    errorMessage: row.error_message ? String(row.error_message) : null,
    createdAt: String(row.created_at),
    updatedAt: String(row.updated_at),
    startedAt: row.started_at ? String(row.started_at) : null,
    completedAt: row.completed_at ? String(row.completed_at) : null,
    projectName: String(row.project_name),
  };
}

function getProjectName(input: { fileName: string; modelId: string }) {
  const extension = path.extname(input.fileName);
  const baseName = path.basename(input.fileName, extension).trim();
  const model = getGenerationModel(input.modelId);
  const readableBaseName = baseName || 'Untitled Asset';

  return model ? `${readableBaseName} // ${model.shortLabel}` : readableBaseName;
}

async function getUserByEmail(email: string) {
  const db = await getDatabaseClient();
  const result = await db.execute({
    sql: 'SELECT * FROM users WHERE lower(email) = lower(?) LIMIT 1',
    args: [email],
  });

  if (result.rows.length === 0) {
    return null;
  }

  const row = result.rows[0] as Record<string, unknown>;
  return {
    profile: mapUserProfile(row),
    passwordHash: String(row.password_hash),
  };
}

export async function getUserBySessionToken(token: string) {
  const db = await getDatabaseClient();
  const result = await db.execute({
    sql: `
      SELECT users.*
      FROM sessions
      JOIN users ON users.id = sessions.user_id
      WHERE sessions.token = ?
        AND datetime(sessions.expires_at) > datetime('now')
      LIMIT 1
    `,
    args: [token],
  });

  if (result.rows.length === 0) {
    return null;
  }

  return mapUserProfile(result.rows[0] as Record<string, unknown>);
}

export async function createSessionRecord(input: { token: string; userId: string; expiresAt: string }) {
  const db = await getDatabaseClient();
  await db.execute({
    sql: `
      INSERT INTO sessions (token, user_id, expires_at)
      VALUES (?, ?, ?)
    `,
    args: [input.token, input.userId, input.expiresAt],
  });
}

export async function deleteSessionRecord(token: string) {
  const db = await getDatabaseClient();
  await db.execute({
    sql: 'DELETE FROM sessions WHERE token = ?',
    args: [token],
  });
}

export async function signupUser(payload: SignupPayload) {
  const existingUser = await getUserByEmail(payload.email);

  if (existingUser) {
    throw new Error('An account with this email already exists.');
  }

  const profile: UserProfile = {
    id: `user-${randomUUID()}`,
    fullName: payload.fullName,
    email: payload.email.toLowerCase(),
    initials: createInitials(payload.fullName),
    ...newUserProfileDefaults,
  };

  await provisionNewUser(profile, hashPassword(payload.password));
  return profile;
}

export async function loginUser(payload: LoginPayload) {
  const user = await getUserByEmail(payload.email);

  if (!user || !verifyPassword(payload.password, user.passwordHash)) {
    return null;
  }

  return user.profile;
}

export async function getShellSummary(userId: string): Promise<ShellSummary> {
  const db = await getDatabaseClient();
  const userResult = await db.execute({
    sql: 'SELECT * FROM users WHERE id = ? LIMIT 1',
    args: [userId],
  });

  return {
    user: mapUserProfile(userResult.rows[0] as Record<string, unknown>),
  };
}

export async function getWorkspaces(userId: string) {
  const db = await getDatabaseClient();
  const result = await db.execute({
    sql: `
      SELECT * FROM workspaces
      WHERE user_id = ?
      ORDER BY is_primary DESC, name ASC
    `,
    args: [userId],
  });
  return result.rows.map((row) => mapWorkspace(row as Record<string, unknown>));
}

export async function getProjects(userId: string, options?: { favoritesOnly?: boolean; recentOnly?: boolean }) {
  const db = await getDatabaseClient();
  const clauses: string[] = ['user_id = ?'];
  const args: Array<string | number> = [userId];

  if (options?.favoritesOnly) {
    clauses.push('is_favorite = 1');
  }

  if (options?.recentOnly) {
    clauses.push('is_recent = 1');
  }

  const result = await db.execute({
    sql: `
      SELECT * FROM projects
      WHERE ${clauses.join(' AND ')}
      ORDER BY datetime(submitted_at) DESC, datetime(created_at) DESC, sort_order ASC, name ASC
    `,
    args,
  });

  return result.rows.map((row) => mapProject(row as Record<string, unknown>));
}

export async function getProjectById(userId: string, projectId: string) {
  const db = await getDatabaseClient();
  const result = await db.execute({
    sql: 'SELECT * FROM projects WHERE id = ? AND user_id = ? LIMIT 1',
    args: [projectId, userId],
  });

  if (result.rows.length === 0) {
    return null;
  }

  return mapProject(result.rows[0] as Record<string, unknown>);
}

export async function getProjectByIdForProcessing(projectId: string) {
  const db = await getDatabaseClient();
  const result = await db.execute({
    sql: 'SELECT * FROM projects WHERE id = ? LIMIT 1',
    args: [projectId],
  });

  if (result.rows.length === 0) {
    return null;
  }

  return mapProject(result.rows[0] as Record<string, unknown>);
}

export async function getSettings(userId: string) {
  const db = await getDatabaseClient();
  await ensureDefaultSettings(userId);
  const result = await db.execute({
    sql: 'SELECT * FROM settings WHERE user_id = ? ORDER BY section_title, label',
    args: [userId],
  });

  const grouped = new Map<string, SettingSection>();

  for (const row of result.rows as Record<string, unknown>[]) {
    const sectionId = String(row.section_id);
    const currentSection = grouped.get(sectionId);

    const item: SettingItem = {
      id: String(row.id),
      key: String(row.id).replace(`${userId}-`, ''),
      label: String(row.label),
      value: String(row.value),
      description: String(row.description),
    };

    if (currentSection) {
      currentSection.items.push(item);
      continue;
    }

    grouped.set(sectionId, {
      id: sectionId,
      title: String(row.section_title),
      description: String(row.section_description),
      items: [item],
    });
  }

  return Array.from(grouped.values());
}

async function ensureDefaultSettings(userId: string) {
  const db = await getDatabaseClient();

  for (const section of defaultSettingSections) {
    for (const item of section.items) {
      await db.execute({
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

export async function updateUserProfileDetails(input: { userId: string; fullName: string }) {
  const db = await getDatabaseClient();
  await db.execute({
    sql: `
      UPDATE users
      SET full_name = ?, initials = ?
      WHERE id = ?
    `,
    args: [input.fullName, createInitials(input.fullName), input.userId],
  });
}

export async function updateUserSettings(input: {
  userId: string;
  updates: Array<{ id: string; value: string }>;
}) {
  if (input.updates.length === 0) {
    return;
  }

  const db = await getDatabaseClient();

  for (const update of input.updates) {
    await db.execute({
      sql: `
        UPDATE settings
        SET value = ?
        WHERE id = ? AND user_id = ?
      `,
      args: [update.value, update.id, input.userId],
    });
  }
}

export async function updateProjectForUser(input: {
  userId: string;
  projectId: string;
  updates: ProjectUpdateInput;
}) {
  const normalized = normalizeProjectUpdateInput(input.updates);
  const currentProject = await getProjectById(input.userId, input.projectId);

  if (!currentProject) {
    return null;
  }

  if (!normalized.name && normalized.isFavorite === undefined) {
    return currentProject;
  }

  const db = await getDatabaseClient();
  const assignments: string[] = [];
  const args: Array<string | number> = [];

  if (normalized.name) {
    assignments.push('name = ?');
    args.push(normalized.name);
  }

  if (normalized.isFavorite !== undefined) {
    assignments.push('is_favorite = ?');
    args.push(normalized.isFavorite ? 1 : 0);
  }

  assignments.push('updated_label = ?');
  args.push(projectRecordDefaults.updatedLabel);
  args.push(input.projectId, input.userId);

  await db.execute({
    sql: `
      UPDATE projects
      SET ${assignments.join(', ')}
      WHERE id = ? AND user_id = ?
    `,
    args,
  });

  if (
    normalized.isFavorite !== undefined
    && normalized.isFavorite !== currentProject.isFavorite
  ) {
    await db.execute({
      sql: `
        UPDATE workspaces
        SET favorite_count = MAX(0, favorite_count + ?)
        WHERE id = ? AND user_id = ?
      `,
      args: [normalized.isFavorite ? 1 : -1, currentProject.workspaceId, input.userId],
    });
  }

  return getProjectById(input.userId, input.projectId);
}

export async function getProjectAssetReferenceCounts(paths: string[]) {
  const uniquePaths = Array.from(new Set(paths.filter(Boolean)));

  if (uniquePaths.length === 0) {
    return new Map<string, number>();
  }

  const db = await getDatabaseClient();
  const counts = new Map<string, number>();

  for (const assetPath of uniquePaths) {
    const result = await db.execute({
      sql: `
        SELECT COUNT(*) AS count
        FROM projects
        WHERE source_image_path = ?
           OR mask_image_path = ?
           OR output_asset_path = ?
      `,
      args: [assetPath, assetPath, assetPath],
    });

    counts.set(assetPath, Number(result.rows[0]?.count ?? 0));
  }

  return counts;
}

export async function deleteProjectForUser(input: { userId: string; projectId: string }) {
  const project = await getProjectById(input.userId, input.projectId);

  if (!project) {
    return null;
  }

  const db = await getDatabaseClient();

  await db.batch(
    [
      {
        sql: 'DELETE FROM generation_jobs WHERE project_id = ? AND user_id = ?',
        args: [input.projectId, input.userId],
      },
      {
        sql: 'DELETE FROM projects WHERE id = ? AND user_id = ?',
        args: [input.projectId, input.userId],
      },
      {
        sql: `
          UPDATE workspaces
          SET project_count = MAX(0, project_count - 1),
              favorite_count = MAX(0, favorite_count - ?),
              updated_label = ?
          WHERE id = ? AND user_id = ?
        `,
        args: [project.isFavorite ? 1 : 0, projectRecordDefaults.workspaceUpdatedLabel, project.workspaceId, input.userId],
      },
    ],
    'write',
  );

  return project;
}

async function getPrimaryWorkspace(userId: string) {
  const db = await getDatabaseClient();
  const workspaceResult = await db.execute({
    sql: 'SELECT * FROM workspaces WHERE user_id = ? ORDER BY is_primary DESC LIMIT 1',
    args: [userId],
  });
  const workspace = workspaceResult.rows[0] as Record<string, unknown> | undefined;

  if (!workspace) {
    throw new Error('Workspace not found for current user.');
  }

  return workspace;
}

export async function createGenerationDraft(input: {
  userId: string;
  modelId: string;
  providerId: string;
  prompt: string;
  sourceImagePath: string;
  maskImagePath?: string | null;
  sourceFileName: string;
  outputFormat: string;
  parameterValues: GenerationParameterValueMap;
}) {
  const db = await getDatabaseClient();
  const workspace = await getPrimaryWorkspace(input.userId);
  const projectId = `project-${Date.now()}`;
  const jobId = `job-${Date.now()}-${randomUUID()}`;
  const submittedAt = new Date().toISOString();
  const model = getGenerationModel(input.modelId);
  const defaults = model ? getModelParameterDefaults(model) : {};

  const requestPayload: GenerationRequestPayload & { sourceImagePath: string; maskImagePath?: string | null } = {
    modelId: input.modelId,
    prompt: input.prompt,
    outputFormat: input.outputFormat,
    parameterValues: { ...defaults, ...input.parameterValues },
    sourceImagePath: input.sourceImagePath,
    maskImagePath: input.maskImagePath ?? null,
  };

  await db.execute({
    sql: `
      INSERT INTO projects (
        id, user_id, workspace_id, workspace_name, name, format, updated_label, tris_label, visual, prompt,
        seed, resolution, creativity, detail_level, tri_count, vert_count, fps, auto_save_label, is_favorite,
        is_recent, sort_order, generation_status, provider_id, model_id, generation_job_id, parameter_values_json,
        source_image_path, mask_image_path, output_asset_path, output_format, error_message, submitted_at, completed_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `,
    args: [
      projectId,
      input.userId,
      String(workspace.id),
      String(workspace.name),
      getProjectName({ fileName: input.sourceFileName, modelId: input.modelId }),
      input.outputFormat.toUpperCase(),
      projectRecordDefaults.updatedLabel,
      'Queued',
      'globe',
      input.prompt,
      String(input.parameterValues.seed ?? defaults.seed ?? 'Auto'),
      String(input.parameterValues['settings.resolution'] ?? defaults['settings.resolution'] ?? 'Default'),
      Number(input.parameterValues['settings.sparseStructure.guidanceStrength'] ?? defaults['settings.sparseStructure.guidanceStrength'] ?? 0),
      String(input.parameterValues['settings.decimationTarget'] ?? defaults['settings.decimationTarget'] ?? 'Default'),
      projectRecordDefaults.triCount,
      projectRecordDefaults.vertCount,
      projectRecordDefaults.fps,
      'Queued for generation',
      0,
      1,
      0,
      'queued',
      input.providerId,
      input.modelId,
      jobId,
      JSON.stringify(requestPayload.parameterValues),
      input.sourceImagePath,
      input.maskImagePath ?? null,
      null,
      input.outputFormat,
      null,
      submittedAt,
      null,
    ],
  });

  await db.execute({
    sql: `
      INSERT INTO generation_jobs (
        id, project_id, user_id, provider_id, model_id, status, provider_task_id,
        request_payload_json, response_payload_json, attempt_count, error_message, created_at, updated_at, started_at, completed_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `,
    args: [
      jobId,
      projectId,
      input.userId,
      input.providerId,
      input.modelId,
      'queued',
      null,
      JSON.stringify(requestPayload),
      null,
      0,
      null,
      submittedAt,
      submittedAt,
      null,
      null,
    ],
  });

  await db.execute({
    sql: `
      UPDATE workspaces
      SET project_count = project_count + 1, updated_label = ?
      WHERE id = ?
    `,
    args: [projectRecordDefaults.workspaceUpdatedLabel, String(workspace.id)],
  });

  return { projectId, jobId };
}

export async function getGenerationJobForUser(userId: string, jobId: string) {
  const db = await getDatabaseClient();
  const result = await db.execute({
    sql: `
      SELECT generation_jobs.*, projects.name AS project_name
      FROM generation_jobs
      JOIN projects ON projects.id = generation_jobs.project_id
      WHERE generation_jobs.id = ? AND generation_jobs.user_id = ?
      LIMIT 1
    `,
    args: [jobId, userId],
  });

  if (result.rows.length === 0) {
    return null;
  }

  return mapGenerationJob(result.rows[0] as Record<string, unknown>);
}

export async function getGenerationJobForProcessing(jobId: string) {
  const db = await getDatabaseClient();
  const result = await db.execute({
    sql: `
      SELECT generation_jobs.*, projects.name AS project_name
      FROM generation_jobs
      JOIN projects ON projects.id = generation_jobs.project_id
      WHERE generation_jobs.id = ?
      LIMIT 1
    `,
    args: [jobId],
  });

  if (result.rows.length === 0) {
    return null;
  }

  return mapGenerationJob(result.rows[0] as Record<string, unknown>);
}

export async function markGenerationJobRunning(jobId: string) {
  const db = await getDatabaseClient();
  const now = new Date().toISOString();

  await db.execute({
    sql: `
      UPDATE generation_jobs
      SET status = 'running', started_at = ?, updated_at = ?
      WHERE id = ?
    `,
    args: [now, now, jobId],
  });

  await db.execute({
    sql: `
      UPDATE projects
      SET generation_status = 'running', updated_label = 'Processing now', tris_label = 'Generating', auto_save_label = 'Generation in progress'
      WHERE generation_job_id = ?
    `,
    args: [jobId],
  });
}

export async function incrementGenerationJobAttempt(jobId: string) {
  const db = await getDatabaseClient();
  const now = new Date().toISOString();

  await db.execute({
    sql: `
      UPDATE generation_jobs
      SET attempt_count = attempt_count + 1,
          updated_at = ?
      WHERE id = ?
    `,
    args: [now, jobId],
  });
}

export async function markGenerationJobProviderPending(input: {
  jobId: string;
  providerTaskId: string;
  responsePayloadJson: string;
}) {
  const db = await getDatabaseClient();
  const now = new Date().toISOString();

  await db.execute({
    sql: `
      UPDATE generation_jobs
      SET status = 'running',
          provider_task_id = ?,
          response_payload_json = ?,
          updated_at = ?
      WHERE id = ?
    `,
    args: [input.providerTaskId, input.responsePayloadJson, now, input.jobId],
  });

  await db.execute({
    sql: `
      UPDATE projects
      SET generation_status = 'running',
          updated_label = 'Processing now',
          tris_label = 'Generating',
          auto_save_label = 'Generation in progress'
      WHERE generation_job_id = ?
    `,
    args: [input.jobId],
  });
}

export async function completeGenerationJob(input: {
  jobId: string;
  providerTaskId: string | null;
  responsePayloadJson: string;
  outputAssetPath: string;
  outputFormat: string;
}) {
  const db = await getDatabaseClient();
  const completedAt = new Date().toISOString();

  await db.execute({
    sql: `
      UPDATE generation_jobs
      SET status = 'succeeded',
          provider_task_id = ?,
          response_payload_json = ?,
          completed_at = ?,
          updated_at = ?
      WHERE id = ?
    `,
    args: [input.providerTaskId, input.responsePayloadJson, completedAt, completedAt, input.jobId],
  });

  await db.execute({
    sql: `
      UPDATE projects
      SET generation_status = 'succeeded',
          output_asset_path = ?,
          output_format = ?,
          format = ?,
          updated_label = 'Ready',
          tris_label = '3D asset ready',
          auto_save_label = 'Saved to workspace',
          error_message = NULL,
          completed_at = ?
      WHERE generation_job_id = ?
    `,
    args: [input.outputAssetPath, input.outputFormat, input.outputFormat.toUpperCase(), completedAt, input.jobId],
  });
}

export async function failGenerationJob(input: {
  jobId: string;
  providerTaskId?: string | null;
  responsePayloadJson?: string | null;
  errorMessage: string;
}) {
  const db = await getDatabaseClient();
  const completedAt = new Date().toISOString();

  await db.execute({
    sql: `
      UPDATE generation_jobs
      SET status = 'failed',
          provider_task_id = ?,
          response_payload_json = ?,
          error_message = ?,
          completed_at = ?,
          updated_at = ?
      WHERE id = ?
    `,
    args: [
      input.providerTaskId ?? null,
      input.responsePayloadJson ?? null,
      input.errorMessage,
      completedAt,
      completedAt,
      input.jobId,
    ],
  });

  await db.execute({
    sql: `
      UPDATE projects
      SET generation_status = 'failed',
          tris_label = 'Generation failed',
          auto_save_label = 'Review error details',
          error_message = ?,
          completed_at = ?
      WHERE generation_job_id = ?
    `,
    args: [input.errorMessage, completedAt, input.jobId],
  });
}

export async function getProjectFilePathForUser(input: {
  userId: string;
  projectId: string;
  kind: 'source' | 'mask' | 'output';
}) {
  const db = await getDatabaseClient();
  const fieldName = input.kind === 'source'
    ? 'source_image_path'
    : input.kind === 'mask'
      ? 'mask_image_path'
      : 'output_asset_path';
  const result = await db.execute({
    sql: `SELECT ${fieldName} AS file_path FROM projects WHERE id = ? AND user_id = ? LIMIT 1`,
    args: [input.projectId, input.userId],
  });

  const row = result.rows[0] as Record<string, unknown> | undefined;
  const filePath = row?.file_path;

  return typeof filePath === 'string' ? filePath : null;
}

export async function getSourceImagePathFromProject(input: { userId: string; projectId: string }) {
  return getProjectFilePathForUser({ ...input, kind: 'source' });
}

export async function getMaskImagePathFromProject(input: { userId: string; projectId: string }) {
  return getProjectFilePathForUser({ ...input, kind: 'mask' });
}
