import { NextResponse } from 'next/server';
import { createUserSession } from '@/lib/auth/session';
import { loginUser } from '@/lib/db/repository';

export async function POST(request: Request) {
  const body = await request.json().catch(() => null) as {
    email?: string;
    password?: string;
  } | null;

  if (!body) {
    return NextResponse.json({ message: 'Invalid login request.' }, { status: 400 });
  }

  const email = body.email?.trim();
  const password = body.password ?? '';

  if (!email || !password) {
    return NextResponse.json({ message: 'Email and password are required.' }, { status: 400 });
  }

  const user = await loginUser({
    email,
    password,
  });

  if (!user) {
    return NextResponse.json({ message: 'Invalid email or password.' }, { status: 401 });
  }

  await createUserSession(user.id);
  return NextResponse.json({ ok: true });
}
