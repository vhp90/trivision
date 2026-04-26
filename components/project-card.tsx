'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Box, Plus, RotateCcw, Star, Trash2 } from 'lucide-react';
import { dashboardContent } from '@/content/site';
import { studioDefaults } from '@/lib/config/app';
import type { ProjectRecord } from '@/lib/db/types';

function getStatusClass(status: ProjectRecord['status']) {
  if (status === 'failed') {
    return 'border-error/50 bg-error/10 text-error';
  }

  if (status === 'succeeded') {
    return 'border-primary/40 bg-primary/10 text-primary';
  }

  return 'border-border-muted bg-surface text-text-main';
}

function AssetFallback() {
  return (
    <div className="relative flex h-full w-full items-center justify-center overflow-hidden bg-[linear-gradient(135deg,rgba(255,170,0,0.08),rgba(255,255,255,0.02)_42%,rgba(0,0,0,0)_42%)]">
      <div className="absolute inset-0 opacity-40 [background-image:linear-gradient(rgba(255,255,255,0.06)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.06)_1px,transparent_1px)] [background-size:28px_28px]" />
      <div className="relative flex h-24 w-24 items-center justify-center border border-border-muted bg-surface shadow-2xl transition-transform duration-500 group-hover:-translate-y-1 group-hover:border-primary/60">
        <div className="absolute -right-3 -top-3 h-10 w-10 border border-primary/30" />
        <div className="absolute -bottom-3 -left-3 h-8 w-8 border border-border-muted bg-background-dark" />
        <Box className="h-9 w-9 text-text-muted transition-colors duration-300 group-hover:text-primary" />
      </div>
    </div>
  );
}

export function ProjectCard({ project }: { project: ProjectRecord }) {
  const router = useRouter();
  const [isMutating, setIsMutating] = useState(false);
  const statusLabel = studioDefaults.jobStatusLabels[project.status];
  const sourcePreviewUrl = project.sourceImagePath ? `/api/projects/${project.id}/asset?kind=source` : null;
  const isActive = project.status === 'queued' || project.status === 'running';

  async function updateProject(updates: { isFavorite?: boolean }) {
    setIsMutating(true);

    try {
      const response = await fetch(`/api/projects/${project.id}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updates),
      });

      if (!response.ok) {
        return;
      }

      router.refresh();
    } finally {
      setIsMutating(false);
    }
  }

  async function deleteProject() {
    if (!window.confirm(`Delete "${project.name}"?`)) {
      return;
    }

    setIsMutating(true);

    try {
      const response = await fetch(`/api/projects/${project.id}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        router.refresh();
      }
    } finally {
      setIsMutating(false);
    }
  }

  async function retryProject() {
    setIsMutating(true);

    try {
      const response = await fetch(`/api/projects/${project.id}/retry`, {
        method: 'POST',
      });
      const payload = await response.json().catch(() => null) as { projectId?: string } | null;

      if (response.ok && payload?.projectId) {
        router.push(`/studio?projectId=${payload.projectId}`);
        router.refresh();
      }
    } finally {
      setIsMutating(false);
    }
  }

  return (
    <article className="group flex w-full flex-col border border-border-muted bg-surface transition-colors hover:border-primary">
      <Link prefetch={false} href={`/studio?projectId=${project.id}`} className="block">
        <div className="relative flex aspect-[4/3] w-full items-center justify-center overflow-hidden border-b border-border-muted bg-background-dark">
          {sourcePreviewUrl ? (
            <>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={sourcePreviewUrl}
                alt={project.name}
                loading="lazy"
                className="absolute inset-0 h-full w-full object-cover transition-transform duration-500 group-hover:scale-[1.03]"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-background-dark/90 via-background-dark/15 to-transparent"></div>
            </>
          ) : (
            <AssetFallback />
          )}
          {project.format ? (
            <div className="absolute top-2 right-2 flex gap-1">
              <span className="bg-background-dark/80 backdrop-blur text-[10px] font-mono text-primary px-1.5 py-0.5 border border-border-muted">{project.format}</span>
            </div>
          ) : null}
          <div className="absolute bottom-2 left-2">
            <span className={`border px-1.5 py-0.5 text-[10px] font-mono ${getStatusClass(project.status)}`}>
              {statusLabel}
            </span>
          </div>
        </div>
        <div className="p-3 flex flex-col gap-1">
          <h3 className="text-sm font-body font-medium text-text-main truncate group-hover:text-primary transition-colors">{project.name}</h3>
          <div className="flex items-center justify-between mt-1">
            <p className="text-[11px] font-mono text-text-muted">{project.updatedLabel}</p>
            <p className="text-[11px] font-mono text-text-muted">{project.trisLabel}</p>
          </div>
        </div>
      </Link>
      <div className="flex items-center justify-between border-t border-border-muted px-2 py-2">
        <button
          type="button"
          disabled={isMutating}
          onClick={() => updateProject({ isFavorite: !project.isFavorite })}
          className={`h-8 w-8 border border-border-muted flex items-center justify-center transition-colors disabled:opacity-50 ${project.isFavorite ? 'text-primary bg-primary/10' : 'text-text-muted hover:text-primary hover:border-primary'}`}
          title={project.isFavorite ? 'Remove from favorites' : 'Add to favorites'}
        >
          <Star className={`h-4 w-4 ${project.isFavorite ? 'fill-current' : ''}`} />
        </button>
        <div className="flex items-center gap-1">
          {project.status === 'failed' ? (
            <button
              type="button"
              disabled={isMutating}
              onClick={retryProject}
              className="h-8 w-8 border border-border-muted flex items-center justify-center text-text-muted hover:text-primary hover:border-primary transition-colors disabled:opacity-50"
              title="Retry generation"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
          ) : null}
          <button
            type="button"
            disabled={isMutating || isActive}
            onClick={deleteProject}
            className="h-8 w-8 border border-border-muted flex items-center justify-center text-text-muted hover:text-error hover:border-error/60 transition-colors disabled:opacity-50"
            title={isActive ? 'Wait for this generation to finish before deleting' : 'Delete generation'}
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>
    </article>
  );
}

export function CreateProjectCard() {
  return (
    <Link prefetch={false} href="/studio" className="group flex min-h-48 w-full flex-col items-center justify-center gap-3 border border-dashed border-border-muted bg-surface/50 transition-all duration-300 hover:border-primary hover:bg-surface-hover/50">
      <div className="flex h-12 w-12 items-center justify-center border border-border-muted bg-surface text-text-muted transition-all duration-300 group-hover:border-primary group-hover:bg-primary group-hover:text-background-dark">
        <Plus className="w-6 h-6" />
      </div>
      <span className="text-sm font-display font-medium text-text-muted group-hover:text-primary transition-colors">{dashboardContent.createCardLabel}</span>
    </Link>
  );
}
