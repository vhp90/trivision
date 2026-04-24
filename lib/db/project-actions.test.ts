import { describe, expect, it } from 'vitest';
import { normalizeProjectUpdateInput } from '@/lib/db/project-actions';

describe('project update normalization', () => {
  it('trims names and keeps favorite changes explicit', () => {
    expect(normalizeProjectUpdateInput({ name: '  Demo Asset  ', isFavorite: true })).toEqual({
      name: 'Demo Asset',
      isFavorite: true,
    });
  });

  it('ignores blank names so update requests cannot erase visible labels', () => {
    expect(normalizeProjectUpdateInput({ name: '   ', isFavorite: false })).toEqual({
      isFavorite: false,
    });
  });
});
