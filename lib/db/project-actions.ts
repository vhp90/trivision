export type ProjectUpdateInput = {
  name?: unknown;
  isFavorite?: unknown;
};

export function normalizeProjectUpdateInput(input: ProjectUpdateInput) {
  const normalized: {
    name?: string;
    isFavorite?: boolean;
  } = {};

  if (typeof input.name === 'string') {
    const name = input.name.trim();

    if (name) {
      normalized.name = name;
    }
  }

  if (typeof input.isFavorite === 'boolean') {
    normalized.isFavorite = input.isFavorite;
  }

  return normalized;
}
