import { samAdapter } from '@/lib/generation/providers/sam';
import { trellisAdapter } from '@/lib/generation/providers/trellis';

export const providerAdapters = {
  [trellisAdapter.modelId]: trellisAdapter,
  [samAdapter.modelId]: samAdapter,
};

export function getProviderAdapter(modelId: string) {
  const adapter = providerAdapters[modelId as keyof typeof providerAdapters];

  if (!adapter) {
    throw new Error(`No provider adapter is configured for model ${modelId}.`);
  }

  return adapter;
}
