'use client';
import { createElement, useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  Box,
  Cuboid,
  Download,
  Grid3X3,
  Image as ImageIcon,
  Lightbulb,
  Menu,
  Play,
  RotateCcw,
  Save,
  ScanSearch,
  Settings,
  Shapes,
  Star,
  Type,
  Trash2,
  Upload,
  Video,
} from 'lucide-react';
import { generationRuntimeDefaults, studioDefaults } from '@/lib/config/app';
import { studioContent } from '@/content/site';
import {
  generationModels,
  getDefaultGenerationModel,
  getGenerationModel,
  getModelParameterDefaults,
  groupModelParameters,
} from '@/lib/generation/registry';
import type {
  GenerationJobStatus,
  GenerationParameterDefinition,
  GenerationParameterValueMap,
} from '@/lib/generation/types';
import { getFriendlyGenerationError } from '@/lib/generation/errors';
import type { ProjectRecord, SettingSection } from '@/lib/db/types';
import type { MobileSamEditor as MobileSamEditorComponent } from '@/components/mobile-sam-editor';

type StudioPageClientProps = {
  project: ProjectRecord | null;
  settings: SettingSection[];
};

type ViewerMode = 'wireframe' | 'solid';
type LightingMode = 'neutral' | 'boosted';

type GenerationStatusResponse = {
  job: {
    id: string;
    status: GenerationJobStatus;
    errorMessage: string | null;
  };
  project: ProjectRecord;
  assets: {
    sourceImageUrl: string | null;
    maskImageUrl: string | null;
    outputAssetUrl: string | null;
  };
};

const defaultModel = getDefaultGenerationModel();

function formatModelName(name: string) {
  return name.toUpperCase();
}

function getStudioModeLabel(model: typeof generationModels[number]) {
  if (model.capabilities.inputKinds.includes('mask')) {
    return 'Masked Image to 3D';
  }

  if (model.capabilities.inputKinds.includes('image')) {
    return 'Image to 3D';
  }

  return '3D Generation';
}

function getProjectStatusLabel(status: GenerationJobStatus) {
  return studioDefaults.jobStatusLabels[status];
}

function getSettingValue(settings: SettingSection[], id: string) {
  return settings
    .flatMap((section) => section.items)
    .find((item) => (item.key ?? item.id) === id)?.value;
}

function getModelFromSettings(settings: SettingSection[]) {
  const preferredModel = getSettingValue(settings, 'default-model');

  if (!preferredModel) {
    return defaultModel;
  }

  return generationModels.find((model) =>
    model.availability === 'enabled'
    && (model.id === preferredModel || model.shortLabel === preferredModel)
  ) ?? defaultModel;
}

function applySettingDefaults(model: typeof generationModels[number], settings: SettingSection[]) {
  const defaults = getModelParameterDefaults(model);
  const nextDefaults: GenerationParameterValueMap = { ...defaults };
  const settingMappings = [
    { settingId: 'default-resolution', parameterKeys: ['settings.resolution'] },
    { settingId: 'default-texture-size', parameterKeys: ['settings.textureSize', 'textureSize'] },
    { settingId: 'default-decimation', parameterKeys: ['settings.decimationTarget', 'simplifyTarget'] },
  ];

  for (const mapping of settingMappings) {
    const settingValue = getSettingValue(settings, mapping.settingId);

    if (!settingValue) {
      continue;
    }

    for (const parameterKey of mapping.parameterKeys) {
      if (parameterKey in defaults) {
        const numericValue = Number(settingValue);
        nextDefaults[parameterKey] = Number.isNaN(numericValue) ? settingValue : numericValue;
        break;
      }
    }
  }

  return nextDefaults;
}

function getInitialModel(project: ProjectRecord | null, settings: SettingSection[]) {
  return (project?.modelId && getGenerationModel(project.modelId)) || getModelFromSettings(settings);
}

function getInitialParameterValues(project: ProjectRecord | null, settings: SettingSection[]) {
  const model = getInitialModel(project, settings);
  return {
    ...applySettingDefaults(model, settings),
    ...project?.parameterValues,
  };
}

function isLightningTrellisModel(modelId: string) {
  return modelId === 'lightning:microsoft-trellis-2@4b';
}

function ParameterField({
  definition,
  value,
  onChange,
}: {
  definition: GenerationParameterDefinition;
  value: string | number | boolean | undefined;
  onChange: (value: string | number | boolean) => void;
}) {
  if (definition.type === 'boolean') {
    return (
      <button
        type="button"
        onClick={() => onChange(!(value === true))}
        title={definition.description}
        className="flex items-center justify-between border border-border-muted bg-background-dark px-3 py-2 text-left hover:border-primary transition-colors"
      >
        <div>
          <div className="text-[12px] text-text-main">{definition.label}</div>
        </div>
        <div className={`w-10 h-5 rounded-full relative transition-colors ${value === true ? 'bg-primary' : 'bg-surface-hover'}`}>
          <div className={`absolute top-0.5 h-4 w-4 rounded-full bg-surface transition-all ${value === true ? 'right-0.5' : 'left-0.5'}`}></div>
        </div>
      </button>
    );
  }

  if (definition.type === 'select') {
    return (
      <label className="flex flex-col gap-2" title={definition.description}>
        <span className="text-[11px] font-mono text-text-muted uppercase tracking-wider">{definition.label}</span>
        <select
          value={String(value ?? definition.defaultValue)}
          onChange={(event) => onChange(Number.isNaN(Number(event.target.value)) ? event.target.value : Number(event.target.value))}
          className="h-10 border border-border-muted bg-background-dark px-3 text-[12px] text-text-main focus:outline-none focus:border-primary"
        >
          {definition.options?.map((option) => (
            <option key={option.value} value={String(option.value)}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
    );
  }

  return (
    <label className="flex flex-col gap-2" title={definition.description}>
      <span className="text-[11px] font-mono text-text-muted uppercase tracking-wider">{definition.label}</span>
      <input
        type="number"
        min={definition.min}
        max={definition.max}
        step={definition.step ?? 1}
        value={typeof value === 'number' ? value : Number(definition.defaultValue)}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-10 border border-border-muted bg-background-dark px-3 text-[12px] text-text-main focus:outline-none focus:border-primary"
      />
    </label>
  );
}

export function StudioPageClient({ project, settings }: StudioPageClientProps) {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [mobileSamEditor, setMobileSamEditor] = useState<typeof MobileSamEditorComponent | null>(null);
  const [currentProject, setCurrentProject] = useState<ProjectRecord | null>(project);
  const [selectedModelId, setSelectedModelId] = useState<string>(getInitialModel(project, settings).id);
  const [prompt, setPrompt] = useState<string>(project?.prompt ?? studioDefaults.emptyPrompt);
  const [parameterValues, setParameterValues] = useState<GenerationParameterValueMap>(getInitialParameterValues(project, settings));
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [processedSourceFile, setProcessedSourceFile] = useState<File | null>(null);
  const [processedSourcePreviewUrl, setProcessedSourcePreviewUrl] = useState<string | null>(null);
  const [maskFile, setMaskFile] = useState<File | null>(null);
  const [maskPreviewUrl, setMaskPreviewUrl] = useState<string | null>(null);
  const [maskOverlayUrl, setMaskOverlayUrl] = useState<string | null>(null);
  const [maskCleared, setMaskCleared] = useState(false);
  const [isPreprocessing, setIsPreprocessing] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isProjectMutating, setIsProjectMutating] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [projectActionMessage, setProjectActionMessage] = useState('');
  const [projectName, setProjectName] = useState(project?.name ?? '');
  const [viewerMode, setViewerMode] = useState<ViewerMode>('solid');
  const [lightingMode, setLightingMode] = useState<LightingMode>('neutral');
  const [viewerRevision, setViewerRevision] = useState(0);

  const selectedModel = useMemo(
    () => getGenerationModel(selectedModelId) ?? defaultModel,
    [selectedModelId],
  );
  const parameterGroups = useMemo(
    () => groupModelParameters(selectedModel),
    [selectedModel],
  );
  const modelName = formatModelName(currentProject?.name ?? selectedModel.shortLabel ?? studioDefaults.emptyModelName);
  const currentStatus = currentProject?.status ?? 'succeeded';
  const currentProjectError = currentProject?.errorMessage
    ? getFriendlyGenerationError(currentProject.errorMessage)
    : '';
  const studioModeLabel = getStudioModeLabel(selectedModel);
  const autoSaveLabel = currentProject?.autoSaveLabel ?? studioDefaults.emptyAutoSaveLabel;
  const promptDisabled = selectedModel.capabilities.promptSupport === 'none';
  const showPromptField = !promptDisabled;
  const requiresLightningPreprocess = isLightningTrellisModel(selectedModel.id);
  const hasPersistedLightningSource = Boolean(!sourceFile && currentProject?.sourceImagePath && currentProject?.modelId === selectedModel.id);
  const disabledModelLabels = generationModels
    .filter((model) => model.availability === 'disabled')
    .map((model) => `${model.shortLabel}: ${model.disabledReason ?? 'Unavailable'}`);
  const persistedSourcePreviewUrl = currentProject?.sourceImagePath ? `/api/projects/${currentProject.id}/asset?kind=source` : null;
  const persistedProcessedPreviewUrl = requiresLightningPreprocess && hasPersistedLightningSource ? persistedSourcePreviewUrl : null;
  const persistedMaskPreviewUrl = !sourceFile && currentProject?.maskImagePath ? `/api/projects/${currentProject.id}/asset?kind=mask` : null;
  const outputAssetUrl = currentProject?.outputAssetPath ? `/api/projects/${currentProject.id}/asset?kind=output` : null;
  const uploadedSourcePreviewUrl = useMemo(
    () => (sourceFile ? URL.createObjectURL(sourceFile) : null),
    [sourceFile],
  );
  const sourcePreviewUrl = uploadedSourcePreviewUrl ?? persistedSourcePreviewUrl;
  const activeProcessedPreviewUrl = processedSourcePreviewUrl ?? persistedProcessedPreviewUrl;
  const activeMaskPreviewUrl = maskPreviewUrl ?? (maskCleared ? null : persistedMaskPreviewUrl);
  const isJobActive = isSubmitting || currentProject?.status === 'queued' || currentProject?.status === 'running';
  const isCurrentProjectActive = currentProject?.status === 'queued' || currentProject?.status === 'running';
  const requiresMask = selectedModel.capabilities.inputKinds.includes('mask');
  const hasMaskForCurrentSource = Boolean(maskFile) || Boolean(!sourceFile && !maskCleared && currentProject?.maskImagePath);
  const exportFileFormat = (currentProject?.outputFormat ?? selectedModel.defaultOutputFormat).toUpperCase();
  const downloadFileName = `${(currentProject?.name ?? selectedModel.shortLabel).replace(/[^a-zA-Z0-9._-]+/g, '-').toLowerCase()}.${exportFileFormat.toLowerCase()}`;
  const downloadHref = outputAssetUrl
    ? `${outputAssetUrl}${outputAssetUrl.includes('?') ? '&' : '?'}download=1&filename=${encodeURIComponent(downloadFileName)}`
    : null;
  const metrics = [
    { label: 'Tris', value: currentProject?.triCount ?? studioContent.viewerMetrics[0].value },
    { label: 'Verts', value: currentProject?.vertCount ?? studioContent.viewerMetrics[1].value },
    { label: 'FPS', value: currentProject?.fps ?? studioContent.viewerMetrics[2].value, colorClassName: 'text-success' },
  ];

  useEffect(() => {
    void import('@google/model-viewer');
  }, []);

  useEffect(() => {
    if (selectedModel.segmentationSupport?.engine !== 'mobile-sam') {
      return;
    }

    let isMounted = true;

    void import('@/components/mobile-sam-editor').then((module) => {
      if (isMounted) {
        setMobileSamEditor(() => module.MobileSamEditor);
      }
    });

    return () => {
      isMounted = false;
    };
  }, [selectedModel.segmentationSupport?.engine]);

  useEffect(() => {
    if (!uploadedSourcePreviewUrl) {
      return undefined;
    }

    return () => {
      URL.revokeObjectURL(uploadedSourcePreviewUrl);
    };
  }, [uploadedSourcePreviewUrl]);

  useEffect(() => {
    if (!processedSourcePreviewUrl) {
      return undefined;
    }

    return () => {
      URL.revokeObjectURL(processedSourcePreviewUrl);
    };
  }, [processedSourcePreviewUrl]);

  useEffect(() => {
    setProjectName(currentProject?.name ?? '');
  }, [currentProject?.id, currentProject?.name]);

  useEffect(() => {
    const activeJobId = currentProject?.generationJobId;

    if (!activeJobId || (currentProject.status !== 'queued' && currentProject.status !== 'running')) {
      return undefined;
    }

    const intervalId = window.setInterval(async () => {
      try {
        const response = await fetch(`/api/generations/${activeJobId}`, { cache: 'no-store' });

        if (!response.ok) {
          setErrorMessage('Unable to refresh generation status. Retrying...');
          return;
        }

        const payload = await response.json() as GenerationStatusResponse;
        setCurrentProject(payload.project);

        if (payload.project.status === 'succeeded' || payload.project.status === 'failed') {
          router.refresh();
        }
      } catch {
        setErrorMessage('Network error while refreshing generation status. Retrying...');
      }
    }, generationRuntimeDefaults.pollIntervalMs);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [currentProject?.generationJobId, currentProject?.status, router]);

  const canGenerate = selectedModel.availability === 'enabled'
    && !isSubmitting
    && !isPreprocessing
    && !(
      currentProject?.status === 'queued'
      || currentProject?.status === 'running'
    )
    && (requiresLightningPreprocess
      ? Boolean(processedSourceFile) || hasPersistedLightningSource
      : Boolean(sourceFile) || Boolean(currentProject?.sourceImagePath))
    && (!requiresMask || hasMaskForCurrentSource)
    && !(promptDisabled && prompt.trim());
  const generateButtonLabel = isSubmitting
    ? 'STARTING JOB'
    : !sourcePreviewUrl
      ? 'UPLOAD IMAGE FIRST'
      : requiresMask && !hasMaskForCurrentSource
        ? 'CREATE MASK FIRST'
        : studioContent.generateLabel;

  const handleModelChange = (modelId: string) => {
    const model = getGenerationModel(modelId);

    if (!model) {
      return;
    }

    setSelectedModelId(model.id);
    setParameterValues(applySettingDefaults(model, settings));
    setErrorMessage('');
    setProcessedSourceFile(null);
    setProcessedSourcePreviewUrl(null);

    if (model.capabilities.promptSupport === 'none') {
      setPrompt('');
    }
  };

  const handleParameterChange = (key: string, value: string | number | boolean) => {
    setParameterValues((currentValues) => ({
      ...currentValues,
      [key]: value,
    }));
  };

  const handleLightningPreprocess = async () => {
    if (!sourceFile) {
      setErrorMessage('Upload an image before running background removal.');
      return;
    }

    setIsPreprocessing(true);
    setErrorMessage('');

    const formData = new FormData();
    formData.append('sourceImage', sourceFile);

    try {
      const response = await fetch('/api/providers/lightning/rembg', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({ message: 'Unable to remove background.' }));
        setProcessedSourceFile(null);
        setProcessedSourcePreviewUrl(null);
        setErrorMessage(payload.message ?? 'Unable to remove background.');
        return;
      }

      const processedBlob = await response.blob();
      const processedFileName = response.headers.get('X-Processed-Filename') || `${sourceFile.name.replace(/\.[^.]+$/, '')}-rembg.png`;
      const nextProcessedFile = new File([processedBlob], processedFileName, {
        type: processedBlob.type || 'image/png',
      });

      setProcessedSourceFile(nextProcessedFile);
      setProcessedSourcePreviewUrl(URL.createObjectURL(nextProcessedFile));
    } catch {
      setProcessedSourceFile(null);
      setProcessedSourcePreviewUrl(null);
      setErrorMessage('Network error while removing the background.');
    } finally {
      setIsPreprocessing(false);
    }
  };

  const handleGenerate = async () => {
    if (!canGenerate) {
      setErrorMessage(requiresMask
        ? 'Select an image and create a mask before generating with this model.'
        : 'Select an image and use a supported model configuration before generating.');
      return;
    }

    const formData = new FormData();
    const effectiveSourceFile = requiresLightningPreprocess ? processedSourceFile : sourceFile;
    formData.append('modelId', selectedModel.id);
    formData.append('prompt', prompt);
    formData.append('outputFormat', selectedModel.defaultOutputFormat);
    formData.append('parameters', JSON.stringify(parameterValues));

    if (effectiveSourceFile) {
      formData.append('sourceImage', effectiveSourceFile);
    } else if (currentProject?.id && (!requiresLightningPreprocess || hasPersistedLightningSource)) {
      formData.append('sourceProjectId', currentProject.id);
    }

    if (maskFile) {
      formData.append('maskImage', maskFile);
    }

    setIsSubmitting(true);
    setErrorMessage('');

    try {
      const response = await fetch('/api/generations', {
        method: 'POST',
        body: formData,
      });

      const payload = await response.json().catch(() => null) as
        | { message?: string; jobId?: string; projectId?: string }
        | null;

      if (!response.ok || !payload?.jobId || !payload?.projectId) {
        setErrorMessage(payload?.message ?? 'Unable to start generation.');
        return;
      }

      const statusResponse = await fetch(`/api/generations/${payload.jobId}`, { cache: 'no-store' });

      if (statusResponse.ok) {
        const statusPayload = await statusResponse.json() as GenerationStatusResponse;
        setCurrentProject(statusPayload.project);
        setMaskCleared(false);
      } else {
        setErrorMessage('Generation started, but status could not be refreshed yet.');
      }

      setSourceFile(null);
      setProcessedSourceFile(null);
      setProcessedSourcePreviewUrl(null);
      setMaskFile(null);
      setMaskPreviewUrl(null);
      setMaskOverlayUrl(null);
      setMaskCleared(false);
      router.replace(`/studio?projectId=${payload.projectId}`);
      router.refresh();
    } catch {
      setErrorMessage('Network error while starting generation.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleProjectUpdate = async (updates: { name?: string; isFavorite?: boolean }) => {
    if (!currentProject) {
      return;
    }

    setIsProjectMutating(true);
    setProjectActionMessage('');

    try {
      const response = await fetch(`/api/projects/${currentProject.id}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updates),
      });
      const payload = await response.json().catch(() => null) as { message?: string; project?: ProjectRecord } | null;

      if (!response.ok || !payload?.project) {
        setProjectActionMessage(payload?.message ?? 'Unable to update this generation.');
        return;
      }

      setCurrentProject(payload.project);
      setProjectActionMessage('Generation updated.');
      router.refresh();
    } catch {
      setProjectActionMessage('Network error while updating this generation.');
    } finally {
      setIsProjectMutating(false);
    }
  };

  const handleRetryProject = async () => {
    if (!currentProject) {
      return;
    }

    setIsProjectMutating(true);
    setProjectActionMessage('');

    try {
      const response = await fetch(`/api/projects/${currentProject.id}/retry`, {
        method: 'POST',
      });
      const payload = await response.json().catch(() => null) as { message?: string; projectId?: string } | null;

      if (!response.ok || !payload?.projectId) {
        setProjectActionMessage(payload?.message ?? 'Unable to retry this generation.');
        return;
      }

      router.replace(`/studio?projectId=${payload.projectId}`);
      router.refresh();
    } catch {
      setProjectActionMessage('Network error while retrying this generation.');
    } finally {
      setIsProjectMutating(false);
    }
  };

  const handleDeleteProject = async () => {
    if (!currentProject || !window.confirm(`Delete "${currentProject.name}"?`)) {
      return;
    }

    setIsProjectMutating(true);
    setProjectActionMessage('');

    try {
      const response = await fetch(`/api/projects/${currentProject.id}`, {
        method: 'DELETE',
      });
      const payload = await response.json().catch(() => null) as { message?: string } | null;

      if (!response.ok) {
        setProjectActionMessage(payload?.message ?? 'Unable to delete this generation.');
        return;
      }

      router.replace('/dashboard');
      router.refresh();
    } catch {
      setProjectActionMessage('Network error while deleting this generation.');
    } finally {
      setIsProjectMutating(false);
    }
  };

  return (
    <div className="h-dvh w-full overflow-x-auto overflow-y-hidden flex flex-col">
      <header className="h-[40px] flex items-center justify-between border-b border-border-muted bg-surface px-4 shrink-0 z-20">
        <div className="flex items-center gap-4">
          <Link href="/dashboard" className="text-text-muted hover:text-text-main transition-colors flex items-center justify-center">
            <Menu className="w-5 h-5" />
          </Link>
          <div className="flex items-center gap-2">
            <Box className="w-4 h-4 text-primary" />
            <h1 className="font-mono text-[13px] font-medium tracking-tight">{modelName}</h1>
          </div>
          <span className="h-4 w-px bg-border-muted mx-2"></span>
          <div className="flex items-center gap-2 text-[11px] font-mono text-text-muted">
            <span className="flex items-center gap-1">
              <span className={`w-1.5 h-1.5 rounded-full ${currentProject?.status === 'failed' ? 'bg-error' : 'bg-success'}`}></span>
              {getProjectStatusLabel(currentStatus)}
            </span>
            <span>•</span>
            <span>{autoSaveLabel}</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div title={studioContent.tooltips.model} className="flex items-center gap-2 border border-border-muted rounded bg-background-dark px-2 py-1">
            <span className="font-mono text-[11px] text-text-main">{selectedModel.shortLabel}</span>
          </div>
          {downloadHref ? (
            <Link
              href={downloadHref}
              download={downloadFileName}
              title={studioContent.tooltips.download}
              className="h-[28px] px-3 border border-primary text-primary font-display text-[12px] font-medium rounded hover:bg-primary/10 transition-colors flex items-center gap-1.5"
            >
              <Download className="w-3.5 h-3.5" />
              {`${studioContent.exportLabel} .${exportFileFormat}`}
            </Link>
          ) : (
            <button type="button" disabled className="h-[28px] px-3 border border-border-muted text-text-muted font-display text-[12px] font-medium rounded flex items-center gap-1.5 disabled:opacity-60">
              <Download className="w-3.5 h-3.5" />
              {`${studioContent.exportLabel} .${exportFileFormat}`}
            </button>
          )}
        </div>
      </header>

      <div className="flex flex-1 min-w-[960px] overflow-hidden relative">
        <aside className="w-[320px] flex flex-col border-r border-border-muted bg-surface shrink-0 z-10 relative">
          <div className="p-4 border-b border-border-muted flex items-center gap-2">
            {showPromptField ? <Type className="w-4 h-4" /> : <ImageIcon className="w-4 h-4" />}
            <h2 className="font-display font-bold text-[14px]">{studioModeLabel}</h2>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-6 pb-[92px]">
            <div className="space-y-2">
              <label className="text-[11px] font-mono text-text-muted uppercase tracking-wider">{studioContent.modelLabel}</label>
              <select
                value={selectedModel.id}
                onChange={(event) => handleModelChange(event.target.value)}
                title={studioContent.tooltips.model}
                className="w-full h-10 border border-border-muted bg-background-dark px-3 text-[13px] text-text-main focus:outline-none focus:border-primary"
              >
                {generationModels.map((model) => (
                  <option key={model.id} value={model.id} disabled={model.availability !== 'enabled'}>
                    {model.label}{model.availability !== 'enabled' ? ` (${studioContent.disabledModelBadge})` : ''}
                  </option>
                ))}
              </select>
              {disabledModelLabels.length > 0 ? (
                <div className="flex flex-wrap gap-2 pt-1">
                  {disabledModelLabels.map((label) => (
                    <span key={label} title={label} className="text-[10px] font-mono text-text-muted border border-border-muted px-2 py-1">
                      {studioContent.disabledModelBadge}
                    </span>
                  ))}
                </div>
              ) : null}
            </div>

            {showPromptField ? (
              <div className="space-y-2">
                <label className="flex justify-between items-center">
                  <span className="text-[11px] font-mono text-text-muted uppercase tracking-wider">{studioContent.promptLabel}</span>
                  <button type="button" onClick={() => setPrompt('')} className="text-[11px] text-primary hover:underline">{studioContent.clearPromptLabel}</button>
                </label>
                <textarea
                  className="w-full h-28 bg-background-dark border border-border-muted rounded p-3 text-[13px] font-body text-text-main placeholder:text-text-muted focus:border-primary focus:ring-1 focus:ring-primary focus:outline-none resize-none transition-all disabled:opacity-60"
                  placeholder={studioContent.promptPlaceholder}
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  disabled={promptDisabled}
                  readOnly={promptDisabled}
                  title={selectedModel.promptHelperText || studioContent.tooltips.prompt}
                />
              </div>
            ) : null}

            <div className="space-y-2">
              <label className="text-[11px] font-mono text-text-muted uppercase tracking-wider">{studioContent.referenceImageLabel}</label>
              <div className="border border-dashed border-border-muted rounded-lg bg-background-dark/50 p-4 text-center">
                {sourcePreviewUrl ? (
                  <div className="space-y-3">
                    <div className="relative aspect-square overflow-hidden border border-border-muted bg-background-dark">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={sourcePreviewUrl}
                        alt="Uploaded source"
                        className="h-full w-full object-cover"
                      />
                    </div>
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      title={studioContent.tooltips.sourceImage}
                      className="w-full h-10 border border-border-muted bg-surface hover:border-primary hover:text-primary transition-colors text-sm font-body text-text-main"
                    >
                      {studioContent.replaceImageLabel}
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    title={studioContent.tooltips.sourceImage}
                    className="w-full flex flex-col items-center justify-center gap-3 py-8 cursor-pointer group"
                  >
                    <div className="w-10 h-10 rounded-full bg-surface-hover flex items-center justify-center group-hover:bg-primary/20 group-hover:text-primary transition-colors">
                      <Upload className="w-5 h-5" />
                    </div>
                    <div>
                      <p className="text-[13px] font-medium mb-1">{studioContent.referenceImageTitle}</p>
                      <p className="text-[11px] text-text-muted font-mono">{studioContent.referenceImageHelp}</p>
                    </div>
                  </button>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  className="hidden"
                  onChange={(event) => {
                    const file = event.target.files?.[0] ?? null;
                    setSourceFile(file);
                    setProcessedSourceFile(null);
                    setProcessedSourcePreviewUrl(null);
                    setMaskFile(null);
                    setMaskPreviewUrl(null);
                    setMaskOverlayUrl(null);
                    setMaskCleared(Boolean(file));
                    if (file) {
                      setErrorMessage('');
                    }
                  }}
                />
              </div>
            </div>

            {requiresLightningPreprocess ? (
              <div className="space-y-3 border border-border-muted bg-background-dark p-4">
                <div>
                  <div className="text-[11px] font-mono text-text-muted uppercase tracking-wider">{studioContent.lightningPrepTitle}</div>
                  <div className="mt-1 text-[12px] text-text-muted">{studioContent.lightningPrepHelp}</div>
                </div>
                <div className="border border-border-muted bg-surface p-3 space-y-3">
                  <div className="aspect-square overflow-hidden border border-border-muted bg-background-dark relative">
                    {activeProcessedPreviewUrl ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={activeProcessedPreviewUrl}
                        alt="Background removed preview"
                        className="h-full w-full object-cover"
                      />
                    ) : (
                      <div className="absolute inset-0 flex items-center justify-center px-4 text-center text-[11px] font-mono text-text-muted">
                        {studioContent.lightningPrepPendingLabel}
                      </div>
                    )}
                  </div>
                  <button
                    type="button"
                    onClick={handleLightningPreprocess}
                    disabled={!sourceFile || isPreprocessing || isSubmitting}
                    className="w-full h-10 border border-border-muted bg-surface hover:border-primary hover:text-primary disabled:opacity-60 disabled:hover:border-border-muted disabled:hover:text-text-main transition-colors text-sm font-body text-text-main"
                  >
                    {isPreprocessing ? studioContent.lightningPrepProcessingLabel : studioContent.lightningPrepAction}
                  </button>
                  <div className="text-[11px] font-mono text-text-muted">
                    {activeProcessedPreviewUrl ? studioContent.lightningPrepReadyLabel : studioContent.lightningPrepPendingLabel}
                  </div>
                </div>
              </div>
            ) : null}

            {selectedModel.segmentationSupport?.engine === 'mobile-sam' ? (
              <div className="space-y-3 border border-border-muted bg-background-dark p-4">
                <div>
                  <div className="text-[11px] font-mono text-text-muted uppercase tracking-wider">{studioContent.segmentationTitle}</div>
                  <div className="mt-1 text-[12px] text-text-muted">{studioContent.segmentationHelp}</div>
                </div>
                {mobileSamEditor ? (
                  createElement(mobileSamEditor, {
                    sourceImageUrl: sourcePreviewUrl,
                    sourceFingerprint: `${selectedModel.id}-${sourcePreviewUrl ?? 'empty'}`,
                    existingMaskUrl: activeMaskPreviewUrl,
                    onMaskChange: (input: {
                      file: File;
                      maskPreviewUrl: string;
                      overlayPreviewUrl: string;
                      selectedPixelCount: number;
                    }) => {
                      setMaskFile(input.file);
                      setMaskPreviewUrl(input.maskPreviewUrl);
                      setMaskOverlayUrl(input.overlayPreviewUrl);
                      setMaskCleared(false);
                      setErrorMessage('');
                    },
                    onMaskClear: () => {
                      setMaskFile(null);
                      setMaskPreviewUrl(null);
                      setMaskOverlayUrl(null);
                      setMaskCleared(true);
                    },
                  })
                ) : (
                  <div className="flex min-h-32 items-center justify-center border border-border-muted bg-background-dark text-[11px] font-mono text-text-muted">
                    Loading MobileSAM tools...
                  </div>
                )}
              </div>
            ) : null}

            <div className="border border-border-muted bg-background-dark p-4">
              <div className="text-[11px] font-mono text-text-muted uppercase tracking-wider mb-2">{studioContent.jobStatusTitle}</div>
              <div className="text-[18px] font-display font-bold text-text-main">{getProjectStatusLabel(currentStatus)}</div>
              {currentProjectError ? (
                <p className="text-[11px] font-mono text-error mt-2">{currentProjectError}</p>
              ) : null}
              {currentProject ? (
                <div className="mt-4 space-y-3">
                  <label className="flex flex-col gap-2">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-text-muted">Asset Name</span>
                    <input
                      value={projectName}
                      onChange={(event) => setProjectName(event.target.value)}
                      className="h-9 border border-border-muted bg-surface px-3 text-[12px] text-text-main focus:border-primary focus:outline-none"
                    />
                  </label>
                  <div className="grid grid-cols-4 gap-2">
                    <button
                      type="button"
                      disabled={isProjectMutating || !projectName.trim() || projectName.trim() === currentProject.name}
                      onClick={() => handleProjectUpdate({ name: projectName })}
                      title="Save name"
                      className="h-9 border border-border-muted text-text-muted hover:text-primary hover:border-primary disabled:opacity-50 transition-colors flex items-center justify-center"
                    >
                      <Save className="h-4 w-4" />
                    </button>
                    <button
                      type="button"
                      disabled={isProjectMutating}
                      onClick={() => handleProjectUpdate({ isFavorite: !currentProject.isFavorite })}
                      title={currentProject.isFavorite ? 'Remove from favorites' : 'Add to favorites'}
                      className={`h-9 border border-border-muted transition-colors flex items-center justify-center disabled:opacity-50 ${currentProject.isFavorite ? 'text-primary bg-primary/10' : 'text-text-muted hover:text-primary hover:border-primary'}`}
                    >
                      <Star className={`h-4 w-4 ${currentProject.isFavorite ? 'fill-current' : ''}`} />
                    </button>
                    <button
                      type="button"
                      disabled={isProjectMutating || isCurrentProjectActive}
                      onClick={handleRetryProject}
                      title="Retry generation"
                      className="h-9 border border-border-muted text-text-muted hover:text-primary hover:border-primary disabled:opacity-50 transition-colors flex items-center justify-center"
                    >
                      <RotateCcw className="h-4 w-4" />
                    </button>
                    <button
                      type="button"
                      disabled={isProjectMutating || isCurrentProjectActive}
                      onClick={handleDeleteProject}
                      title="Delete generation"
                      className="h-9 border border-border-muted text-text-muted hover:text-error hover:border-error/60 disabled:opacity-50 transition-colors flex items-center justify-center"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                  {projectActionMessage ? (
                    <p className="text-[11px] font-mono text-text-muted">{projectActionMessage}</p>
                  ) : null}
                </div>
              ) : null}
            </div>
          </div>

          <div className="absolute bottom-0 left-0 w-full p-4 bg-surface border-t border-border-muted z-20">
            <button
              type="button"
              onClick={handleGenerate}
              disabled={!canGenerate}
              className="w-full h-[40px] border border-primary bg-primary hover:bg-primary-hover disabled:border-border-muted disabled:bg-surface-hover disabled:text-text-muted disabled:hover:bg-surface-hover text-background-dark font-display font-bold text-[13px] rounded tracking-wide transition-colors flex items-center justify-center gap-2 shadow-[0_0_15px_rgba(245,165,36,0.15)]"
            >
              <Play className="w-4 h-4 fill-current" />
              {generateButtonLabel}
            </button>
            {errorMessage ? (
              <p className="mt-3 text-[11px] font-mono text-error">{errorMessage}</p>
            ) : null}
          </div>
        </aside>

        <main className="flex-1 relative bg-background-dark overflow-hidden group">
          <div className="absolute top-4 left-4 flex gap-2 z-10">
            <button
              type="button"
              onClick={() => setViewerMode('wireframe')}
              className={`w-8 h-8 border rounded flex items-center justify-center transition-colors backdrop-blur-sm ${viewerMode === 'wireframe' ? 'bg-surface border-primary text-primary' : 'bg-surface/80 border-border-muted hover:bg-surface hover:border-text-muted'}`}
              title={studioContent.tooltips.wireframe}
            >
              <Grid3X3 className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={() => setViewerMode('solid')}
              className={`w-8 h-8 border rounded flex items-center justify-center transition-colors backdrop-blur-sm ${viewerMode === 'solid' ? 'bg-surface border-primary text-primary' : 'bg-surface/80 border-border-muted hover:bg-surface hover:border-text-muted'}`}
              title={studioContent.tooltips.solid}
            >
              <Cuboid className="w-4 h-4" />
            </button>
          </div>

          <div className="absolute top-4 right-4 flex flex-col gap-2 z-10">
            <button
              type="button"
              onClick={() => setViewerRevision((value) => value + 1)}
              className="w-8 h-8 bg-surface/80 border border-border-muted rounded flex items-center justify-center hover:bg-surface hover:border-text-muted transition-colors backdrop-blur-sm"
              title={studioContent.tooltips.cameraReset}
            >
              <Video className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={() => setLightingMode((value) => value === 'neutral' ? 'boosted' : 'neutral')}
              className={`w-8 h-8 border rounded flex items-center justify-center transition-colors backdrop-blur-sm ${lightingMode === 'boosted' ? 'bg-surface border-primary text-primary' : 'bg-surface/80 border-border-muted hover:bg-surface hover:border-text-muted'}`}
              title={studioContent.tooltips.lighting}
            >
              <Lightbulb className="w-4 h-4" />
            </button>
          </div>

          {viewerMode === 'wireframe' ? (
            <div className="absolute inset-0 pointer-events-none opacity-20">
              <div className="absolute inset-0 bg-grid-pattern"></div>
            </div>
          ) : null}

          <div className="absolute inset-0 flex items-center justify-center">
            {outputAssetUrl && currentProject?.status === 'succeeded' ? (
              createElement('model-viewer', {
                key: `viewer-${currentProject.id}-${viewerRevision}-${viewerMode}-${lightingMode}`,
                src: outputAssetUrl,
                alt: currentProject.name,
                cameraControls: true,
                autoplay: true,
                environmentImage: generationRuntimeDefaults.viewerEnvironmentImage,
                exposure: lightingMode === 'boosted' ? '1.35' : generationRuntimeDefaults.viewerExposure,
                shadowIntensity: '1',
                touchAction: 'pan-y',
                className: `w-full h-full ${viewerMode === 'wireframe' ? 'opacity-55 contrast-125 saturate-0' : ''}`,
              })
            ) : (
              <div className="w-[420px] max-w-[80%] border border-border-muted bg-surface p-8 text-center space-y-3">
                <div className="w-14 h-14 rounded-full bg-background-dark border border-border-muted mx-auto flex items-center justify-center text-primary">
                  <Shapes className="w-7 h-7" />
                </div>
                <h3 className="font-display text-xl font-bold text-text-main">{studioContent.emptyViewerLabel}</h3>
                <p className="text-sm text-text-muted leading-6">{studioContent.emptyViewerHelp}</p>
              </div>
            )}
          </div>

          {isJobActive ? (
            <div className="absolute inset-0 z-20 flex items-center justify-center bg-background-dark/72 backdrop-blur-sm">
              <div className="flex flex-col items-center gap-6 text-center">
                <div className="relative h-52 w-52">
                  <svg
                    viewBox="0 0 220 220"
                    className="absolute inset-0 h-full w-full text-primary opacity-70 animate-spin"
                    style={{ animationDuration: '18s' }}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1"
                  >
                    <circle cx="110" cy="110" r="84" strokeOpacity="0.25" />
                    <path d="M110 24 L184 67 L184 153 L110 196 L36 153 L36 67 Z" />
                    <path d="M110 52 L160 82 L160 138 L110 168 L60 138 L60 82 Z" strokeOpacity="0.7" />
                    <path d="M110 24 L110 196" strokeOpacity="0.35" />
                    <path d="M36 67 L184 153" strokeOpacity="0.25" />
                    <path d="M184 67 L36 153" strokeOpacity="0.25" />
                  </svg>
                  <svg
                    viewBox="0 0 220 220"
                    className="absolute inset-[20%] h-[60%] w-[60%] text-primary/80"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.25"
                  >
                    <circle cx="110" cy="110" r="38" strokeDasharray="4 5" className="animate-spin" style={{ animationDuration: '6s' }} />
                    <path d="M110 70 L144 110 L110 150 L76 110 Z" />
                  </svg>
                </div>
                <div className="space-y-2">
                  <p className="font-display text-2xl font-bold text-text-main">{studioContent.loaderTitle}</p>
                  <p className="font-mono text-[11px] uppercase tracking-[0.25em] text-text-muted">{studioContent.loaderSubtitle}</p>
                </div>
              </div>
            </div>
          ) : null}

          <div className="absolute bottom-4 left-4 font-mono text-[10px] text-text-muted flex gap-4 bg-surface/50 px-2 py-1 rounded border border-border-muted/50 backdrop-blur-sm z-10">
            {metrics.map((metric) => (
              <span key={metric.label}>
                {metric.label}: <span className={'colorClassName' in metric ? metric.colorClassName : 'text-text-main'}>{metric.value}</span>
              </span>
            ))}
          </div>
        </main>

        <aside className="w-[320px] flex flex-col border-l border-border-muted bg-surface shrink-0 z-10 overflow-y-auto">
          <div className="p-3 border-b border-border-muted bg-surface-hover sticky top-0 z-10">
            <h2 className="font-display font-bold text-[13px] text-text-main">{studioContent.propertiesTitle}</h2>
          </div>

          <div className="divide-y divide-border-muted">
            <div className="p-4 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-[12px] font-medium flex items-center gap-2">
                  <ImageIcon className="w-4 h-4 text-text-muted" />
                  {studioContent.sourcePreviewTitle}
                </span>
              </div>
              <div className="border border-border-muted bg-background-dark aspect-square overflow-hidden relative">
                {sourcePreviewUrl ? (
                  <>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={sourcePreviewUrl}
                      alt="Source preview"
                      className="h-full w-full object-cover"
                    />
                    {maskOverlayUrl ? (
                      <>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={maskOverlayUrl}
                          alt=""
                          className="absolute inset-0 h-full w-full object-cover"
                        />
                      </>
                    ) : null}
                  </>
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center text-[11px] font-mono text-text-muted">
                    Awaiting source image
                  </div>
                )}
              </div>
            </div>

            {requiresLightningPreprocess ? (
              <div className="p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-[12px] font-medium flex items-center gap-2">
                    <ImageIcon className="w-4 h-4 text-text-muted" />
                    {studioContent.lightningPrepPreviewTitle}
                  </span>
                </div>
                <div className="border border-border-muted bg-background-dark aspect-square overflow-hidden relative">
                  {activeProcessedPreviewUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={activeProcessedPreviewUrl}
                      alt="Processed preview"
                      className="h-full w-full object-cover"
                    />
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center text-[11px] font-mono text-text-muted">
                      Run background removal to prepare the image
                    </div>
                  )}
                </div>
              </div>
            ) : null}

            {selectedModel.segmentationSupport?.engine === 'mobile-sam' ? (
              <div className="p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-[12px] font-medium flex items-center gap-2">
                    <ScanSearch className="w-4 h-4 text-text-muted" />
                    {studioContent.maskPreviewTitle}
                  </span>
                </div>
                <div className="border border-border-muted bg-background-dark aspect-square overflow-hidden relative">
                  {activeMaskPreviewUrl ? (
                    <>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={activeMaskPreviewUrl}
                        alt="Mask preview"
                        className="h-full w-full object-cover"
                      />
                    </>
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center text-[11px] font-mono text-text-muted">
                      Create a mask to enable SAM 3D generation
                    </div>
                  )}
                </div>
              </div>
            ) : null}

            {parameterGroups.length > 0 ? (
              <div className="p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-[12px] font-medium flex items-center gap-2">
                    <Settings className="w-4 h-4 text-text-muted" />
                    {studioContent.generationParametersTitle}
                  </span>
                </div>

                {parameterGroups.map((group) => (
                  <div key={group.title} className="space-y-3 border border-border-muted bg-background-dark p-3">
                    <h3 className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">{group.title}</h3>
                    <div className="space-y-3">
                      {group.parameters.map((definition) => (
                        <ParameterField
                          key={definition.key}
                          definition={definition}
                          value={parameterValues[definition.key]}
                          onChange={(value) => handleParameterChange(definition.key, value)}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        </aside>
      </div>
    </div>
  );
}
