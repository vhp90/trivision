import { NextResponse } from 'next/server';
import { getAuthenticatedUser } from '@/lib/auth/session';
import { updateUserProfileDetails } from '@/lib/db/repository';

export async function POST(request: Request) {
  const user = await getAuthenticatedUser();

  if (!user) {
    return NextResponse.json({ message: 'Unauthorized.' }, { status: 401 });
  }

  const body = await request.json().catch(() => null) as {
    fullName?: string;
  } | null;
  const fullName = body?.fullName?.trim() ?? '';

  if (!fullName) {
    return NextResponse.json({ message: 'Name is required.' }, { status: 400 });
  }

  await updateUserProfileDetails({
    userId: user.id,
    fullName,
  });

  return NextResponse.json({ ok: true });
}
