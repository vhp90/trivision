'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ArrowRight, Lock, Mail, UserRound, ShieldAlert } from 'lucide-react';

type AuthCredentialFormProps = {
  mode: 'login' | 'signup';
  emailLabel: string;
  emailPlaceholder: string;
  passwordLabel: string;
  passwordPlaceholder: string;
  submitLabel: string;
  secondaryPrompt: string;
  secondaryHref: string;
  secondaryLabel: string;
  fullNameLabel?: string;
  fullNamePlaceholder?: string;
};

export function AuthCredentialForm({
  mode,
  emailLabel,
  emailPlaceholder,
  passwordLabel,
  passwordPlaceholder,
  submitLabel,
  secondaryPrompt,
  secondaryHref,
  secondaryLabel,
  fullNameLabel = 'Name',
  fullNamePlaceholder = 'Your full name',
}: AuthCredentialFormProps) {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [status, setStatus] = useState<'idle' | 'submitting' | 'error'>('idle');
  const [message, setMessage] = useState('');

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setStatus('submitting');
    setMessage('');

    try {
      const response = await fetch(mode === 'login' ? '/api/auth/login' : '/api/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(
          mode === 'login'
            ? { email, password }
            : { fullName, email, password },
        ),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({ message: 'Unable to complete request.' }));
        setStatus('error');
        setMessage(payload.message ?? 'Unable to complete request.');
        return;
      }

      window.location.assign('/dashboard');
    } catch {
      setStatus('error');
      setMessage('Network error. Check your connection and try again.');
    }
  };

  return (
    <>
      <form className="flex flex-col gap-5" onSubmit={handleSubmit}>
        {mode === 'signup' ? (
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-mono text-text-muted uppercase tracking-wider" htmlFor="full-name">{fullNameLabel}</label>
            <div className="relative">
              <input
                id="full-name"
                type="text"
                required
                autoComplete="name"
                value={fullName}
                onChange={(event) => setFullName(event.target.value)}
                placeholder={fullNamePlaceholder}
                className="w-full h-10 bg-background-dark border border-border-muted text-text-main px-3 text-sm font-body placeholder:text-border-muted focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
              />
              <UserRound className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted w-4 h-4" />
            </div>
          </div>
        ) : null}

        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-mono text-text-muted uppercase tracking-wider" htmlFor="email">{emailLabel}</label>
          <div className="relative">
            <input
              id="email"
              type="email"
              required
              autoComplete="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder={emailPlaceholder}
              className="w-full h-10 bg-background-dark border border-border-muted text-text-main px-3 text-sm font-body placeholder:text-border-muted focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
            />
            <Mail className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted w-4 h-4" />
          </div>
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-mono text-text-muted uppercase tracking-wider" htmlFor="password">{passwordLabel}</label>
          <div className="relative">
            <input
              id="password"
              type="password"
              required
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder={passwordPlaceholder}
              className={`w-full h-10 bg-background-dark border ${status === 'error' ? 'border-error focus:border-error focus:ring-error' : 'border-border-muted focus:border-primary focus:ring-primary'} text-text-main px-3 text-sm font-body placeholder:text-border-muted focus:outline-none focus:ring-1 transition-all`}
            />
            <Lock className={`absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 ${status === 'error' ? 'text-error' : 'text-text-muted'}`} />
          </div>
        </div>

        {status === 'error' ? (
          <p className="text-[11px] font-mono text-error mt-1 flex items-center gap-1">
            <ShieldAlert className="w-3.5 h-3.5" />
            {message}
          </p>
        ) : null}

        <button
          type="submit"
          disabled={status === 'submitting'}
          className="mt-2 w-full h-12 bg-primary hover:bg-primary-hover disabled:opacity-70 disabled:hover:bg-primary text-background-dark font-display font-bold text-[13px] uppercase tracking-wider transition-colors flex items-center justify-center gap-2 group"
        >
          <span>{status === 'submitting' ? 'Processing' : submitLabel}</span>
          <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
        </button>
      </form>

      <div className="mt-10 pt-6 border-t border-border-muted text-center">
        <p className="text-[13px] text-text-muted font-body">
          {secondaryPrompt}
          <Link href={secondaryHref} className="text-text-main hover:text-primary transition-colors font-medium border-b border-border-muted hover:border-primary pb-0.5 ml-1">
            {secondaryLabel}
          </Link>
        </p>
      </div>
    </>
  );
}
