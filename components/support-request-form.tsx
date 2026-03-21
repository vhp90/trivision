'use client';

import { useState } from 'react';
import { ArrowRight, CheckCircle2, Mail, ShieldAlert } from 'lucide-react';

type SupportRequestFormProps = {
  defaultEmail: string;
  emailPlaceholder: string;
  noteLabel: string;
  notePlaceholder: string;
  primaryLabel: string;
  successLabel: string;
};

export function SupportRequestForm({
  defaultEmail,
  emailPlaceholder,
  noteLabel,
  notePlaceholder,
  primaryLabel,
  successLabel,
}: SupportRequestFormProps) {
  const [email, setEmail] = useState(defaultEmail);
  const [note, setNote] = useState('');
  const [status, setStatus] = useState<'idle' | 'submitting' | 'submitted' | 'error'>('idle');
  const [message, setMessage] = useState('');

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setStatus('submitting');
    setMessage('');

    const response = await fetch('/api/support-requests', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email,
        note,
      }),
    });

    if (!response.ok) {
      setStatus('error');
      setMessage('SIGNAL_REJECTED // try again');
      return;
    }

    setStatus('submitted');
    setMessage(successLabel);
    setNote('');
  };

  return (
    <form className="flex flex-col gap-5" onSubmit={handleSubmit}>
      <div className="flex flex-col gap-1.5">
        <label className="text-xs font-mono text-text-muted uppercase tracking-wider" htmlFor="support-email">Email Address</label>
        <div className="relative">
          <input
            id="support-email"
            type="email"
            required
            autoComplete="email"
            placeholder={emailPlaceholder}
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            className="w-full h-10 bg-background-dark border border-border-muted text-text-main px-3 text-sm font-body placeholder:text-border-muted focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
          />
          <Mail className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted w-4 h-4" />
        </div>
      </div>

      <div className="flex flex-col gap-1.5">
        <label className="text-xs font-mono text-text-muted uppercase tracking-wider" htmlFor="support-note">{noteLabel}</label>
        <textarea
          id="support-note"
          required
          value={note}
          onChange={(event) => setNote(event.target.value)}
          placeholder={notePlaceholder}
          className="w-full min-h-32 bg-background-dark border border-border-muted text-text-main px-3 py-3 text-sm font-body placeholder:text-border-muted focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all resize-none"
        />
      </div>

      <button
        type="submit"
        disabled={status === 'submitting'}
        className="mt-2 w-full h-12 bg-primary hover:bg-primary-hover disabled:opacity-70 disabled:hover:bg-primary text-background-dark font-display font-bold text-[13px] uppercase tracking-wider transition-colors flex items-center justify-center gap-2 group"
      >
        <span>{status === 'submitting' ? 'Processing' : primaryLabel}</span>
        <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
      </button>

      {status === 'submitted' ? (
        <p className="text-[11px] font-mono text-success mt-1 flex items-center gap-1">
          <CheckCircle2 className="w-3.5 h-3.5" />
          {message}
        </p>
      ) : null}

      {status === 'error' ? (
        <p className="text-[11px] font-mono text-error mt-1 flex items-center gap-1">
          <ShieldAlert className="w-3.5 h-3.5" />
          {message}
        </p>
      ) : null}
    </form>
  );
}
