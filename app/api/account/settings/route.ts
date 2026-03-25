import { NextResponse } from 'next/server';
import { getAuthenticatedUser } from '@/lib/auth/session';
import { updateUserSettings } from '@/lib/db/repository';

export async function POST(request: Request) {
  const user = await getAuthenticatedUser();

  if (!user) {
    return NextResponse.json({ message: 'Unauthorized.' }, { status: 401 });
  }

  const body = await request.json().catch(() => null) as {
    updates?: Array<{ id?: string; value?: string }>;
  } | null;
  const updates = (body?.updates ?? [])
    .map((update) => ({
      id: update.id?.trim() ?? '',
      value: update.value?.trim() ?? '',
    }))
    .filter((update) => update.id.length > 0);

  if (updates.length === 0) {
    return NextResponse.json({ message: 'At least one setting update is required.' }, { status: 400 });
  }

  await updateUserSettings({
    userId: user.id,
    updates,
  });

  return NextResponse.json({ ok: true });
}
