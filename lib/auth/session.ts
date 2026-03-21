import { randomUUID } from 'node:crypto';
import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';
import type { UserProfile } from '@/lib/db/types';
import {
  createSessionRecord,
  deleteSessionRecord,
  getUserBySessionToken,
} from '@/lib/db/repository';

const SESSION_COOKIE = 'trivision_session';

export async function createUserSession(userId: string) {
  const token = randomUUID();
  const expiresAt = new Date(Date.now() + 1000 * 60 * 60 * 24 * 30).toISOString();

  await createSessionRecord({
    token,
    userId,
    expiresAt,
  });

  const cookieStore = await cookies();
  cookieStore.set(SESSION_COOKIE, token, {
    httpOnly: true,
    sameSite: 'lax',
    secure: process.env.NODE_ENV === 'production',
    path: '/',
    expires: new Date(expiresAt),
  });
}

export async function clearUserSession() {
  const cookieStore = await cookies();
  const token = cookieStore.get(SESSION_COOKIE)?.value;

  if (token) {
    await deleteSessionRecord(token);
  }

  cookieStore.delete(SESSION_COOKIE);
}

export async function getAuthenticatedUser() {
  const cookieStore = await cookies();
  const token = cookieStore.get(SESSION_COOKIE)?.value;

  if (!token) {
    return null;
  }

  return getUserBySessionToken(token);
}

export async function requireAuthenticatedUser(): Promise<UserProfile> {
  const user = await getAuthenticatedUser();

  if (!user) {
    redirect('/login');
  }

  return user;
}
