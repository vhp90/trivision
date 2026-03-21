import { NextResponse } from 'next/server';
import { createSupportRequest } from '@/lib/db/repository';

export async function POST(request: Request) {
  const body = await request.json() as {
    email?: string;
    note?: string;
  };

  const email = body.email?.trim();
  const note = body.note?.trim();

  if (!email || !note) {
    return NextResponse.json({ message: 'Missing required fields.' }, { status: 400 });
  }

  const payload = await createSupportRequest({
    email,
    note,
  });

  return NextResponse.json(payload, { status: 201 });
}
