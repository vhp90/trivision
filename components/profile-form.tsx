'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import type { UserProfile } from '@/lib/db/types';

type ProfileFormProps = {
  user: UserProfile;
};

export function ProfileForm({ user }: ProfileFormProps) {
  const router = useRouter();
  const [fullName, setFullName] = useState(user.fullName);
  const [status, setStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [message, setMessage] = useState('');

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setStatus('saving');
    setMessage('');

    try {
      const response = await fetch('/api/account/profile', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ fullName }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({ message: 'Unable to save profile.' }));
        setStatus('error');
        setMessage(payload.message ?? 'Unable to save profile.');
        return;
      }

      setStatus('saved');
      setMessage('Profile updated.');
      router.refresh();
    } catch {
      setStatus('error');
      setMessage('Network error. Check your connection and try again.');
    }
  }

  return (
    <form onSubmit={handleSubmit} className="bg-surface border border-border-muted p-6 space-y-5">
      <div>
        <h2 className="text-xl font-display font-bold text-text-main">Profile Details</h2>
        <p className="mt-2 text-sm text-text-muted leading-6">
          Update the account identity shown across the local workspace shell.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <label className="flex flex-col gap-2">
          <span className="text-[11px] font-mono uppercase tracking-[0.2em] text-text-muted">Full Name</span>
          <input
            value={fullName}
            onChange={(event) => setFullName(event.target.value)}
            className="h-11 border border-border-muted bg-background-dark px-3 text-sm text-text-main focus:border-primary focus:outline-none"
          />
        </label>

        <label className="flex flex-col gap-2">
          <span className="text-[11px] font-mono uppercase tracking-[0.2em] text-text-muted">Email</span>
          <input
            value={user.email}
            disabled
            className="h-11 border border-border-muted bg-background-dark px-3 text-sm text-text-muted"
          />
        </label>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="border border-border-muted bg-background-dark p-4">
          <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Role</div>
          <div className="mt-2 text-sm text-text-main">{user.roleLabel}</div>
        </div>
        <div className="border border-border-muted bg-background-dark p-4">
          <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Region</div>
          <div className="mt-2 text-sm text-text-main">{user.region}</div>
        </div>
        <div className="border border-border-muted bg-background-dark p-4">
          <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Session</div>
          <div className="mt-2 text-sm text-primary">{user.sessionLabel}</div>
        </div>
      </div>

      <div className="flex items-center justify-between gap-4 border-t border-border-muted pt-5">
        <p className={`text-[11px] font-mono ${status === 'error' ? 'text-error' : 'text-text-muted'}`}>
          {message || 'Changes are stored in the local MVP workspace database.'}
        </p>
        <button
          type="submit"
          disabled={status === 'saving'}
          className="h-10 px-4 bg-primary text-background-dark text-sm font-display font-bold hover:bg-primary-hover disabled:opacity-70 transition-colors"
        >
          {status === 'saving' ? 'Saving' : 'Save Profile'}
        </button>
      </div>
    </form>
  );
}
