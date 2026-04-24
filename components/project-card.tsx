'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Car, Factory, Globe, Lightbulb, Plus, RotateCcw, Star, Trash2 } from 'lucide-react';
import { dashboardContent } from '@/content/site';
import { studioDefaults } from '@/lib/config/app';
import type { ProjectRecord } from '@/lib/db/types';

const projectVisuals = {
  factory: {
    Icon: Factory,
    containerClassName: 'w-3/4 h-3/4 bg-surface-hover shadow-2xl rotate-12 group-hover:rotate-0 transition-transform duration-700 ease-out border border-border-muted flex items-center justify-center relative',
    overlayClassName: 'absolute inset-0 bg-primary/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500',
  },
  lightbulb: {
    Icon: Lightbulb,
    containerClassName: 'w-2/3 h-4/5 bg-surface-hover skew-x-6 group-hover:skew-x-0 transition-transform duration-700 ease-out border border-border-muted flex items-center justify-center relative',
    overlayClassName: 'absolute inset-0 bg-primary/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500',
  },
  globe: {
    Icon: Globe,
    containerClassName: 'w-3/4 h-3/4 rounded-full bg-surface-hover scale-90 group-hover:scale-100 transition-transform duration-700 ease-out border border-border-muted flex items-center justify-center relative shadow-inner',
    overlayClassName: 'absolute inset-0 rounded-full bg-primary/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500',
  },
  car: {
    Icon: Car,
    containerClassName: 'w-4/5 h-2/3 bg-surface-hover -rotate-6 group-hover:rotate-0 transition-transform duration-700 ease-out border border-border-muted flex items-center justify-center relative',
    overlayClassName: 'absolute inset-0 bg-primary/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500',
  },
} as const;

export function ProjectCard({ project }: { project: ProjectRecord }) {
  const router = useRouter();
  const [isMutating, setIsMutating] = useState(false);
  const visual = projectVisuals[project.visual];
  const Icon = visual.Icon;
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
    <article className="group flex flex-col w-full bg-surface border border-border-muted hover:border-primary transition-colors">
      <Link prefetch={false} href={`/studio?projectId=${project.id}`} className="block">
        <div className="w-full aspect-square relative overflow-hidden bg-background-dark border-b border-border-muted p-4 flex items-center justify-center">
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
            <div className={visual.containerClassName}>
              <div className={visual.overlayClassName}></div>
              <Icon className="w-10 h-10 text-text-muted group-hover:text-primary transition-colors duration-300 z-10" />
            </div>
          )}
          {project.format ? (
            <div className="absolute top-2 right-2 flex gap-1">
              <span className="bg-background-dark/80 backdrop-blur text-[10px] font-mono text-primary px-1.5 py-0.5 border border-border-muted">{project.format}</span>
            </div>
          ) : null}
          <div className="absolute bottom-2 left-2">
            <span className={`text-[10px] font-mono px-1.5 py-0.5 border ${project.status === 'failed' ? 'text-error border-error/40 bg-background-dark/80' : project.status === 'succeeded' ? 'text-primary border-primary/40 bg-background-dark/80' : 'text-text-main border-border-muted bg-background-dark/80'}`}>
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
    <Link prefetch={false} href="/studio" className="group w-full aspect-square bg-surface/50 border border-dashed border-border-muted flex flex-col items-center justify-center gap-3 hover:border-primary hover:bg-surface-hover/50 transition-all duration-300">
      <div className="w-12 h-12 rounded-full bg-surface border border-border-muted flex items-center justify-center group-hover:bg-primary group-hover:border-primary group-hover:text-background-dark text-text-muted transition-all duration-300">
        <Plus className="w-6 h-6" />
      </div>
      <span className="text-sm font-display font-medium text-text-muted group-hover:text-primary transition-colors">{dashboardContent.createCardLabel}</span>
    </Link>
  );
}
