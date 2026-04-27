'use client';

import { useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { generationModels } from '@/lib/generation/registry';
import type { SettingSection } from '@/lib/db/types';

type SettingsPreferencesFormProps = {
  sections: SettingSection[];
};

const hiddenSettingKeys = new Set([
  'autosave-interval',
  'notification-mode',
  'review-visibility',
  'session-retention',
]);
const sectionCopyOverrides: Record<string, { title: string; description: string }> = {
  preferences: {
    title: 'Workspace Preferences',
    description: 'Default values applied when you start a new generation session.',
  },
  generation: {
    title: 'Generation Defaults',
    description: 'Baseline model settings used to keep new assets consistent.',
  },
  collaboration: {
    title: 'Runtime Policies',
    description: 'Workspace defaults for predictable generation and export behavior.',
  },
  runtime: {
    title: 'Runtime Policies',
    description: 'Workspace defaults for predictable generation and export behavior.',
  },
};

const itemDescriptionOverrides: Record<string, string> = {
  'default-resolution': 'Baseline voxel resolution for new generations.',
  'default-texture-size': 'Baseline texture size for image-to-3D jobs.',
  'default-decimation': 'Baseline mesh simplification target for output assets.',
};

const selectOptions: Record<string, string[]> = {
  'default-model': generationModels
    .filter((model) => model.availability === 'enabled')
    .map((model) => model.shortLabel),
  'default-output-format': ['GLB'],
  'default-resolution': ['512', '1024', '1536'],
  'default-texture-size': ['1024', '2048', '3072', '4096'],
  'default-decimation': ['100000', '250000', '500000', '750000', '1000000'],
};

export function SettingsPreferencesForm({ sections }: SettingsPreferencesFormProps) {
  const router = useRouter();
  const visibleSections = useMemo(
    () => sections
      .map((section) => ({
        ...section,
        ...(sectionCopyOverrides[section.id] ?? {}),
        items: section.items
          .filter((item) => !hiddenSettingKeys.has(item.key ?? item.id))
          .map((item) => ({
            ...item,
            description: itemDescriptionOverrides[item.key ?? item.id] ?? item.description,
          })),
      }))
      .filter((section) => section.items.length > 0),
    [sections],
  );
  const [values, setValues] = useState<Record<string, string>>(
    Object.fromEntries(
      visibleSections.flatMap((section) => section.items.map((item) => [item.id, item.value])),
    ),
  );
  const [status, setStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [message, setMessage] = useState('');

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setStatus('saving');
    setMessage('');

    const updates = visibleSections.flatMap((section) =>
      section.items.map((item) => ({
        id: item.id,
        value: values[item.id] ?? item.value,
      })),
    );

    try {
      const response = await fetch('/api/account/settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ updates }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({ message: 'Unable to save settings.' }));
        setStatus('error');
        setMessage(payload.message ?? 'Unable to save settings.');
        return;
      }

      setStatus('saved');
      setMessage('Settings saved.');
      router.refresh();
    } catch {
      setStatus('error');
      setMessage('Network error. Check your connection and try again.');
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {visibleSections.map((section) => (
        <div key={section.id} className="bg-surface border border-border-muted p-6 space-y-5">
          <div>
            <h2 className="text-xl font-display font-bold text-text-main">{section.title}</h2>
            <p className="mt-2 text-sm text-text-muted leading-6">{section.description}</p>
          </div>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {section.items.map((item) => {
              const settingKey = item.key ?? item.id;
              const options = selectOptions[settingKey];

              return (
                <label key={item.id} className="flex flex-col gap-2">
                  <span className="text-[11px] font-mono uppercase tracking-[0.2em] text-text-muted">{item.label}</span>
                  {options ? (
                    <select
                      value={values[item.id] ?? item.value}
                      onChange={(event) => setValues((current) => ({ ...current, [item.id]: event.target.value }))}
                      className="h-11 border border-border-muted bg-background-dark px-3 text-sm text-text-main focus:border-primary focus:outline-none"
                    >
                      {options.map((option) => (
                        <option key={option} value={option}>{option}</option>
                      ))}
                    </select>
                  ) : (
                    <input
                      value={values[item.id] ?? item.value}
                      onChange={(event) => setValues((current) => ({ ...current, [item.id]: event.target.value }))}
                      className="h-11 border border-border-muted bg-background-dark px-3 text-sm text-text-main focus:border-primary focus:outline-none"
                    />
                  )}
                  <span className="text-xs text-text-muted leading-5">{item.description}</span>
                </label>
              );
            })}
          </div>
        </div>
      ))}

      <div className="flex items-center justify-between gap-4 border border-border-muted bg-surface p-4">
        <p className={`text-[11px] font-mono ${status === 'error' ? 'text-error' : 'text-text-muted'}`}>
          {message || 'These values are saved as your default studio preferences.'}
        </p>
        <button
          type="submit"
          disabled={status === 'saving'}
          className="h-10 px-4 bg-primary text-background-dark text-sm font-display font-bold hover:bg-primary-hover disabled:opacity-70 transition-colors"
        >
          {status === 'saving' ? 'Saving' : 'Save Settings'}
        </button>
      </div>
    </form>
  );
}
