import { NextResponse } from 'next/server';
import { createUserSession } from '@/lib/auth/session';
import { signupUser } from '@/lib/db/repository';

export async function POST(request: Request) {
  const body = await request.json() as {
    fullName?: string;
    email?: string;
    password?: string;
  };

  const fullName = body.fullName?.trim();
  const email = body.email?.trim();
  const password = body.password ?? '';

  if (!fullName || !email || !password) {
    return NextResponse.json({ message: 'Name, email, and password are required.' }, { status: 400 });
  }

  if (password.length < 8) {
    return NextResponse.json({ message: 'Password must be at least 8 characters.' }, { status: 400 });
  }

  try {
    const user = await signupUser({
      fullName,
      email,
      password,
    });

    await createUserSession(user.id);
    return NextResponse.json({ ok: true }, { status: 201 });
  } catch (error) {
    return NextResponse.json(
      { message: error instanceof Error ? error.message : 'Unable to create account.' },
      { status: 409 },
    );
  }
}
