'use client';

import { useEffect, useRef, useState, type MouseEvent, type PointerEvent } from 'react';
import { Circle, Eraser, MinusCircle, MousePointerClick, RefreshCcw, ScanSearch } from 'lucide-react';
import { createMobileSamEmbedding, predictMobileSamMask, type MobileSamEmbedding } from '@/lib/segmentation/mobile-sam';
import type { MobileSamBoxPrompt, MobileSamPromptMode, MobileSamPromptState } from '@/lib/segmentation/types';

type MobileSamEditorProps = {
  sourceImageUrl: string | null;
  sourceFingerprint: string;
  existingMaskUrl: string | null;
  onMaskChange: (input: {
    file: File;
    maskPreviewUrl: string;
    overlayPreviewUrl: string;
    selectedPixelCount: number;
  }) => void;
  onMaskClear: () => void;
};

function normalizeBox(box: MobileSamBoxPrompt) {
  return {
    x0: Math.min(box.x0, box.x1),
    y0: Math.min(box.y0, box.y1),
    x1: Math.max(box.x0, box.x1),
    y1: Math.max(box.y0, box.y1),
  };
}

export function MobileSamEditor({
  sourceImageUrl,
  sourceFingerprint,
  existingMaskUrl,
  onMaskChange,
  onMaskClear,
}: MobileSamEditorProps) {
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [promptMode, setPromptMode] = useState<MobileSamPromptMode>('positive-point');
  const [promptState, setPromptState] = useState<MobileSamPromptState>({ points: [], box: null });
  const [isEmbedding, setIsEmbedding] = useState(false);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [embedding, setEmbedding] = useState<MobileSamEmbedding | null>(null);
  const [overlayPreviewUrl, setOverlayPreviewUrl] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState('Add a click or drag a box to prepare a mask.');
  const [dragOrigin, setDragOrigin] = useState<{ x: number; y: number } | null>(null);
  const [draftBox, setDraftBox] = useState<MobileSamBoxPrompt | null>(null);

  useEffect(() => {
    setPromptState({ points: [], box: null });
    setOverlayPreviewUrl(null);
    setEmbedding(null);
    setStatusMessage('Add a click or drag a box to prepare a mask.');
    setDraftBox(null);
    setDragOrigin(null);
  }, [sourceFingerprint]);

  const hasPrompt = promptState.points.length > 0 || Boolean(promptState.box);
  const activeBox = draftBox ?? promptState.box;

  function clearMaskAndPrompts() {
    setPromptState({ points: [], box: null });
    setOverlayPreviewUrl(null);
    setDraftBox(null);
    setDragOrigin(null);
    setStatusMessage('Segmentation cleared.');
    onMaskClear();
  }

  function resolveImageCoordinates(clientX: number, clientY: number) {
    const image = imageRef.current;

    if (!image) {
      return null;
    }

    const rect = image.getBoundingClientRect();
    const clampedX = Math.max(0, Math.min(rect.width, clientX - rect.left));
    const clampedY = Math.max(0, Math.min(rect.height, clientY - rect.top));

    return {
      x: (clampedX / rect.width) * image.naturalWidth,
      y: (clampedY / rect.height) * image.naturalHeight,
      displayX: clampedX,
      displayY: clampedY,
      displayWidth: rect.width,
      displayHeight: rect.height,
    };
  }

  async function ensureEmbedding() {
    if (!sourceImageUrl) {
      throw new Error('Upload an image before creating a mask.');
    }

    if (embedding) {
      return embedding;
    }

    setIsEmbedding(true);
    setStatusMessage('Preparing MobileSAM embedding...');

    try {
      const nextEmbedding = await createMobileSamEmbedding(sourceImageUrl);
      setEmbedding(nextEmbedding);
      setStatusMessage('Embedding ready. Create a mask when your prompts look right.');
      return nextEmbedding;
    } finally {
      setIsEmbedding(false);
    }
  }

  async function applyMask() {
    if (!sourceImageUrl) {
      setStatusMessage('Upload an image before creating a mask.');
      return;
    }

    if (!hasPrompt) {
      setStatusMessage('Add at least one click or box prompt first.');
      return;
    }

    setIsSegmenting(true);
    setStatusMessage('Generating MobileSAM mask...');

    try {
      const currentEmbedding = await ensureEmbedding();
      const result = await predictMobileSamMask(currentEmbedding, promptState);
      const file = new File([result.maskBlob], 'mobilesam-mask.png', { type: 'image/png' });
      setOverlayPreviewUrl(result.overlayDataUrl);
      setStatusMessage(`Mask ready. ${result.selectedPixelCount.toLocaleString()} pixels selected.`);
      onMaskChange({
        file,
        maskPreviewUrl: result.maskDataUrl,
        overlayPreviewUrl: result.overlayDataUrl,
        selectedPixelCount: result.selectedPixelCount,
      });
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : 'Unable to generate the MobileSAM mask.');
    } finally {
      setIsSegmenting(false);
    }
  }

  function handleImageClick(event: MouseEvent<HTMLDivElement>) {
    if (promptMode === 'box') {
      return;
    }

    const coordinates = resolveImageCoordinates(event.clientX, event.clientY);

    if (!coordinates) {
      return;
    }

    setOverlayPreviewUrl(null);
    onMaskClear();
    setStatusMessage('Prompt updated. Create a mask when ready.');
    setPromptState((current) => ({
      ...current,
      points: [
        ...current.points,
        {
          x: coordinates.x,
          y: coordinates.y,
          label: promptMode === 'negative-point' ? 0 : 1,
        },
      ],
    }));
  }

  function handlePointerDown(event: PointerEvent<HTMLDivElement>) {
    if (promptMode !== 'box') {
      return;
    }

    const coordinates = resolveImageCoordinates(event.clientX, event.clientY);

    if (!coordinates) {
      return;
    }

    setDraftBox({
      x0: coordinates.x,
      y0: coordinates.y,
      x1: coordinates.x,
      y1: coordinates.y,
    });
    setDragOrigin({ x: coordinates.x, y: coordinates.y });
  }

  function handlePointerMove(event: PointerEvent<HTMLDivElement>) {
    if (promptMode !== 'box' || !dragOrigin) {
      return;
    }

    const coordinates = resolveImageCoordinates(event.clientX, event.clientY);

    if (!coordinates) {
      return;
    }

    setDraftBox({
      x0: dragOrigin.x,
      y0: dragOrigin.y,
      x1: coordinates.x,
      y1: coordinates.y,
    });
  }

  function handlePointerUp(event: PointerEvent<HTMLDivElement>) {
    if (promptMode !== 'box' || !dragOrigin) {
      return;
    }

    const coordinates = resolveImageCoordinates(event.clientX, event.clientY);
    setDragOrigin(null);

    if (!coordinates) {
      setDraftBox(null);
      return;
    }

    const box = normalizeBox({
      x0: dragOrigin.x,
      y0: dragOrigin.y,
      x1: coordinates.x,
      y1: coordinates.y,
    });

    if (Math.abs(box.x1 - box.x0) < 8 || Math.abs(box.y1 - box.y0) < 8) {
      setDraftBox(null);
      setStatusMessage('Draw a larger box to create a prompt.');
      return;
    }

    setOverlayPreviewUrl(null);
    onMaskClear();
    setDraftBox(null);
    setPromptState((current) => ({
      ...current,
      box,
    }));
    setStatusMessage('Box prompt updated. Create a mask when ready.');
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-[12px] font-medium text-text-main">
          <ScanSearch className="h-4 w-4 text-primary" />
          MobileSAM Mask
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setPromptMode('positive-point')}
            title="Add positive clicks to mark the object."
            className={`h-8 w-8 border rounded flex items-center justify-center transition-colors ${promptMode === 'positive-point' ? 'border-primary text-primary bg-primary/10' : 'border-border-muted text-text-muted hover:text-text-main hover:border-text-muted'}`}
          >
            <Circle className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => setPromptMode('negative-point')}
            title="Add negative clicks to exclude regions."
            className={`h-8 w-8 border rounded flex items-center justify-center transition-colors ${promptMode === 'negative-point' ? 'border-primary text-primary bg-primary/10' : 'border-border-muted text-text-muted hover:text-text-main hover:border-text-muted'}`}
          >
            <MinusCircle className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => setPromptMode('box')}
            title="Drag a box around the object."
            className={`h-8 w-8 border rounded flex items-center justify-center transition-colors ${promptMode === 'box' ? 'border-primary text-primary bg-primary/10' : 'border-border-muted text-text-muted hover:text-text-main hover:border-text-muted'}`}
          >
            <MousePointerClick className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={clearMaskAndPrompts}
            title="Clear prompts and remove the current mask."
            className="h-8 w-8 border border-border-muted rounded flex items-center justify-center text-text-muted hover:text-text-main hover:border-text-muted transition-colors"
          >
            <Eraser className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="border border-border-muted bg-background-dark p-2">
        {sourceImageUrl ? (
          <div
            className="relative w-full cursor-crosshair"
            onClick={handleImageClick}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={() => {
              if (dragOrigin) {
                setDragOrigin(null);
                setDraftBox(null);
              }
            }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              ref={imageRef}
              src={sourceImageUrl}
              alt="Segmentation source"
              className="block w-full h-auto select-none"
              draggable={false}
            />
            {overlayPreviewUrl ? (
              <>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={overlayPreviewUrl}
                  alt=""
                  className="absolute inset-0 h-full w-full object-cover pointer-events-none"
                />
              </>
            ) : null}
            <div className="absolute inset-0 pointer-events-none">
              {promptState.points.map((point, index) => {
                const image = imageRef.current;
                const rect = image?.getBoundingClientRect();

                if (!image || !rect || rect.width === 0 || rect.height === 0) {
                  return null;
                }

                return (
                  <span
                    key={`${point.x}-${point.y}-${index}`}
                    className={`absolute h-3 w-3 rounded-full border border-background-dark -translate-x-1/2 -translate-y-1/2 ${point.label === 1 ? 'bg-primary' : 'bg-error'}`}
                    style={{
                      left: `${(point.x / image.naturalWidth) * 100}%`,
                      top: `${(point.y / image.naturalHeight) * 100}%`,
                    }}
                  />
                );
              })}
              {activeBox ? (
                <span
                  className="absolute border border-primary/90 bg-primary/10"
                  style={{
                    left: `${(normalizeBox(activeBox).x0 / (imageRef.current?.naturalWidth || 1)) * 100}%`,
                    top: `${(normalizeBox(activeBox).y0 / (imageRef.current?.naturalHeight || 1)) * 100}%`,
                    width: `${((normalizeBox(activeBox).x1 - normalizeBox(activeBox).x0) / (imageRef.current?.naturalWidth || 1)) * 100}%`,
                    height: `${((normalizeBox(activeBox).y1 - normalizeBox(activeBox).y0) / (imageRef.current?.naturalHeight || 1)) * 100}%`,
                  }}
                />
              ) : null}
            </div>
          </div>
        ) : (
          <div className="flex aspect-square items-center justify-center text-[11px] font-mono text-text-muted">
            Upload an image to start segmenting.
          </div>
        )}
      </div>

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={applyMask}
          disabled={!sourceImageUrl || !hasPrompt || isEmbedding || isSegmenting}
          className="h-9 flex-1 border border-primary text-primary rounded font-medium text-[12px] hover:bg-primary/10 disabled:opacity-60 disabled:hover:bg-transparent transition-colors"
        >
          {isEmbedding || isSegmenting ? 'Building Mask' : 'Create Mask'}
        </button>
        <button
          type="button"
          onClick={() => setEmbedding(null)}
          disabled={!sourceImageUrl}
          title="Rebuild the MobileSAM embedding for the current image."
          className="h-9 w-9 border border-border-muted rounded flex items-center justify-center text-text-muted hover:text-text-main hover:border-text-muted disabled:opacity-60 transition-colors"
        >
          <RefreshCcw className="h-4 w-4" />
        </button>
      </div>

      <p className="text-[11px] font-mono text-text-muted">{statusMessage}</p>

      {existingMaskUrl ? (
        <div className="space-y-2">
          <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Current Mask</div>
          <div className="border border-border-muted bg-background-dark p-2">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={existingMaskUrl} alt="Current mask" className="block w-full h-auto" />
          </div>
        </div>
      ) : null}
    </div>
  );
}
