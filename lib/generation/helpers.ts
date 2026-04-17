import { randomUUID } from 'node:crypto';
import type { GenerationParameterValueMap } from '@/lib/generation/types';

export function createGenerationTaskId() {
  return randomUUID();
}

export function setNestedValue(
  target: Record<string, unknown>,
  path: string,
  value: unknown,
) {
  const segments = path.split('.');
  let cursor: Record<string, unknown> = target;

  for (const segment of segments.slice(0, -1)) {
    const current = cursor[segment];

    if (current && typeof current === 'object' && !Array.isArray(current)) {
      cursor = current as Record<string, unknown>;
      continue;
    }

    const next: Record<string, unknown> = {};
    cursor[segment] = next;
    cursor = next;
  }

  cursor[segments[segments.length - 1]] = value;
}

export function buildSettingsObject(parameterValues: GenerationParameterValueMap) {
  const settings: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(parameterValues)) {
    if (!key.startsWith('settings.')) {
      continue;
    }

    setNestedValue(settings, key.replace('settings.', ''), value);
  }

  return settings;
}

export function getNumberParameter(
  parameterValues: GenerationParameterValueMap,
  key: string,
) {
  const value = parameterValues[key];
  return typeof value === 'number' ? value : null;
}

export function getBooleanParameter(
  parameterValues: GenerationParameterValueMap,
  key: string,
) {
  const value = parameterValues[key];
  return typeof value === 'boolean' ? value : null;
}

export function getStringParameter(
  parameterValues: GenerationParameterValueMap,
  key: string,
) {
  const value = parameterValues[key];
  return typeof value === 'string' ? value : null;
}

export function sleep(durationMs: number) {
  return new Promise((resolve) => setTimeout(resolve, durationMs));
}
