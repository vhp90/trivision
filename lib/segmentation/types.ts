export type MobileSamPromptMode = 'positive-point' | 'negative-point' | 'box';

export type MobileSamPointPrompt = {
  x: number;
  y: number;
  label: 0 | 1;
};

export type MobileSamBoxPrompt = {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
};

export type MobileSamPromptState = {
  points: MobileSamPointPrompt[];
  box: MobileSamBoxPrompt | null;
};

export type MobileSamImageScale = {
  width: number;
  height: number;
  samScale: number;
};

export type MobileSamMaskResult = {
  maskBlob: Blob;
  maskDataUrl: string;
  overlayDataUrl: string;
  selectedPixelCount: number;
};
