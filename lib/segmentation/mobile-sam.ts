'use client';

import { mobileSamRuntimeDefaults } from '@/lib/config/app';
import type {
  MobileSamBoxPrompt,
  MobileSamImageScale,
  MobileSamMaskResult,
  MobileSamPromptState,
} from '@/lib/segmentation/types';

type OrtTensorLike = {
  dims: number[];
  data: unknown;
};

type OrtModule = {
  env: {
    wasm: {
      wasmPaths: string;
      numThreads: number;
    };
  };
  Tensor: new (
    type: string,
    data: Float32Array,
    dims: number[],
  ) => OrtTensorLike;
  InferenceSession: {
    create: (path: string) => Promise<OrtSession>;
  };
};

type OrtSession = {
  run: (feeds: Record<string, unknown>) => Promise<Record<string, OrtTensorLike>>;
};

export type MobileSamEmbedding = {
  embedding: OrtTensorLike;
  imageScale: MobileSamImageScale;
};

let ortModulePromise: Promise<OrtModule> | null = null;
let encoderSessionPromise: Promise<OrtSession> | null = null;
let decoderSessionPromise: Promise<OrtSession> | null = null;

function getTargetSize(width: number, height: number) {
  const samScale = mobileSamRuntimeDefaults.targetLongestSide / Math.max(width, height);
  return {
    width: Math.max(1, Math.round(width * samScale)),
    height: Math.max(1, Math.round(height * samScale)),
    samScale,
  };
}

async function getOrt() {
  if (!ortModulePromise) {
    ortModulePromise = import('onnxruntime-web').then((ort) => {
      const runtime = ort as unknown as OrtModule;
      runtime.env.wasm.wasmPaths = mobileSamRuntimeDefaults.wasmBasePath;
      runtime.env.wasm.numThreads = 1;
      return runtime;
    });
  }

  return ortModulePromise;
}

async function getEncoderSession() {
  if (!encoderSessionPromise) {
    encoderSessionPromise = getOrt().then((ort) => ort.InferenceSession.create(mobileSamRuntimeDefaults.encoderModelPath));
  }

  return encoderSessionPromise;
}

async function getDecoderSession() {
  if (!decoderSessionPromise) {
    decoderSessionPromise = getOrt().then((ort) => ort.InferenceSession.create(mobileSamRuntimeDefaults.decoderModelPath));
  }

  return decoderSessionPromise;
}

function createCanvas(width: number, height: number) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Unable to load the source image for segmentation.'));
    image.src = src;
  });
}

function createEncoderInputTensor(
  ort: OrtModule,
  image: HTMLImageElement,
) {
  const resized = getTargetSize(image.naturalWidth || image.width, image.naturalHeight || image.height);
  const canvas = createCanvas(resized.width, resized.height);
  const context = canvas.getContext('2d');

  if (!context) {
    throw new Error('Canvas rendering is unavailable in this browser.');
  }

  context.drawImage(image, 0, 0, resized.width, resized.height);
  const pixels = context.getImageData(0, 0, resized.width, resized.height).data;
  const channelSize = resized.width * resized.height;
  const input = new Float32Array(channelSize * 3);

  for (let pixelIndex = 0; pixelIndex < channelSize; pixelIndex += 1) {
    const offset = pixelIndex * 4;
    const inputOffset = pixelIndex * 3;
    input[inputOffset] = pixels[offset];
    input[inputOffset + 1] = pixels[offset + 1];
    input[inputOffset + 2] = pixels[offset + 2];
  }

  return {
    tensor: new ort.Tensor('float32', input, [resized.height, resized.width, 3]),
    imageScale: {
      width: image.naturalWidth || image.width,
      height: image.naturalHeight || image.height,
      samScale: resized.samScale,
    },
  };
}

function clampBox(box: MobileSamBoxPrompt, width: number, height: number): MobileSamBoxPrompt {
  const x0 = Math.max(0, Math.min(width, Math.min(box.x0, box.x1)));
  const y0 = Math.max(0, Math.min(height, Math.min(box.y0, box.y1)));
  const x1 = Math.max(0, Math.min(width, Math.max(box.x0, box.x1)));
  const y1 = Math.max(0, Math.min(height, Math.max(box.y0, box.y1)));

  return { x0, y0, x1, y1 };
}

function buildPromptTensors(
  ort: OrtModule,
  prompt: MobileSamPromptState,
  embedding: MobileSamEmbedding,
) {
  const clicksLength = prompt.points.length;
  const hasBox = Boolean(prompt.box);
  const pointCount = clicksLength + (hasBox ? 2 : 1);
  const pointCoords = new Float32Array(pointCount * 2);
  const pointLabels = new Float32Array(pointCount);

  for (let index = 0; index < prompt.points.length; index += 1) {
    const point = prompt.points[index];
    pointCoords[index * 2] = point.x * embedding.imageScale.samScale;
    pointCoords[index * 2 + 1] = point.y * embedding.imageScale.samScale;
    pointLabels[index] = point.label;
  }

  if (prompt.box) {
    const box = clampBox(prompt.box, embedding.imageScale.width, embedding.imageScale.height);
    const boxIndex = prompt.points.length;
    pointCoords[boxIndex * 2] = box.x0 * embedding.imageScale.samScale;
    pointCoords[boxIndex * 2 + 1] = box.y0 * embedding.imageScale.samScale;
    pointLabels[boxIndex] = 2;

    pointCoords[(boxIndex + 1) * 2] = box.x1 * embedding.imageScale.samScale;
    pointCoords[(boxIndex + 1) * 2 + 1] = box.y1 * embedding.imageScale.samScale;
    pointLabels[boxIndex + 1] = 3;
  } else {
    pointCoords[prompt.points.length * 2] = 0;
    pointCoords[prompt.points.length * 2 + 1] = 0;
    pointLabels[prompt.points.length] = -1;
  }

  return {
    point_coords: new ort.Tensor('float32', pointCoords, [1, pointCount, 2]),
    point_labels: new ort.Tensor('float32', pointLabels, [1, pointCount]),
    orig_im_size: new ort.Tensor('float32', new Float32Array([embedding.imageScale.height, embedding.imageScale.width]), [2]),
    mask_input: new ort.Tensor('float32', new Float32Array(256 * 256), [1, 1, 256, 256]),
    has_mask_input: new ort.Tensor('float32', new Float32Array([0]), [1]),
  };
}

function createMaskImages(
  mask: Float32Array,
  width: number,
  height: number,
) {
  const maskCanvas = createCanvas(width, height);
  const overlayCanvas = createCanvas(width, height);
  const maskContext = maskCanvas.getContext('2d');
  const overlayContext = overlayCanvas.getContext('2d');

  if (!maskContext || !overlayContext) {
    throw new Error('Canvas rendering is unavailable in this browser.');
  }

  const maskPixels = new Uint8ClampedArray(width * height * 4);
  const overlayPixels = new Uint8ClampedArray(width * height * 4);
  let selectedPixelCount = 0;

  for (let index = 0; index < width * height; index += 1) {
    const selected = mask[index] > 0;
    const offset = index * 4;

    if (selected) {
      selectedPixelCount += 1;
      maskPixels[offset] = 255;
      maskPixels[offset + 1] = 255;
      maskPixels[offset + 2] = 255;
      maskPixels[offset + 3] = 255;

      overlayPixels[offset] = 245;
      overlayPixels[offset + 1] = 165;
      overlayPixels[offset + 2] = 36;
      overlayPixels[offset + 3] = 160;
      continue;
    }

    maskPixels[offset] = 0;
    maskPixels[offset + 1] = 0;
    maskPixels[offset + 2] = 0;
    maskPixels[offset + 3] = 0;

    overlayPixels[offset] = 0;
    overlayPixels[offset + 1] = 0;
    overlayPixels[offset + 2] = 0;
    overlayPixels[offset + 3] = 0;
  }

  maskContext.putImageData(new ImageData(maskPixels, width, height), 0, 0);
  overlayContext.putImageData(new ImageData(overlayPixels, width, height), 0, 0);

  return {
    maskCanvas,
    overlayCanvas,
    selectedPixelCount,
  };
}

function canvasToBlob(canvas: HTMLCanvasElement) {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error('Unable to serialize the generated mask.'));
        return;
      }

      resolve(blob);
    }, 'image/png');
  });
}

export async function createMobileSamEmbedding(sourceImageUrl: string) {
  const [ort, encoderSession, image] = await Promise.all([
    getOrt(),
    getEncoderSession(),
    loadImage(sourceImageUrl),
  ]);
  const encoderInput = createEncoderInputTensor(ort, image);
  const result = await encoderSession.run({
    input_image: encoderInput.tensor,
  });
  const embedding = result.image_embeddings;

  if (!embedding) {
    throw new Error('MobileSAM did not return image embeddings.');
  }

  return {
    embedding,
    imageScale: encoderInput.imageScale,
  } satisfies MobileSamEmbedding;
}

export async function predictMobileSamMask(
  embedding: MobileSamEmbedding,
  prompt: MobileSamPromptState,
) {
  if (prompt.points.length === 0 && !prompt.box) {
    throw new Error('Add at least one point or box prompt before creating a mask.');
  }

  const [ort, decoderSession] = await Promise.all([
    getOrt(),
    getDecoderSession(),
  ]);
  const feeds = {
    image_embeddings: embedding.embedding,
    ...buildPromptTensors(ort, prompt, embedding),
  };
  const result = await decoderSession.run(feeds);
  const masks = result.masks;

  if (!masks || !('data' in masks) || !Array.isArray(masks.dims) || masks.dims.length < 4) {
    throw new Error('MobileSAM returned an invalid mask tensor.');
  }

  const width = Number(masks.dims[masks.dims.length - 1]);
  const height = Number(masks.dims[masks.dims.length - 2]);
  const tensorData = masks.data instanceof Float32Array
    ? masks.data
    : Float32Array.from(Array.from(masks.data as ArrayLike<number>));
  const { maskCanvas, overlayCanvas, selectedPixelCount } = createMaskImages(tensorData, width, height);
  const maskBlob = await canvasToBlob(maskCanvas);

  return {
    maskBlob,
    maskDataUrl: maskCanvas.toDataURL('image/png'),
    overlayDataUrl: overlayCanvas.toDataURL('image/png'),
    selectedPixelCount,
  } satisfies MobileSamMaskResult;
}
