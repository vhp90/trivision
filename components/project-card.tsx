import Link from 'next/link';
import { Car, Factory, Globe, Lightbulb, Plus } from 'lucide-react';
import { dashboardContent } from '@/content/site';
import { studioDefaults } from '@/lib/config/app';
import type { ProjectRecord } from '@/lib/db/types';

const projectVisuals = {
  factory: {
    Icon: Factory,
    containerClassName: 'w-3/4 h-3/4 bg-gradient-to-br from-[#1a1a1c] to-[#2a2a2d] shadow-2xl rotate-12 group-hover:rotate-0 transition-transform duration-700 ease-out border border-border-muted flex items-center justify-center relative',
    overlayClassName: 'absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(245,165,36,0.1)_0%,transparent_70%)] opacity-0 group-hover:opacity-100 transition-opacity duration-500',
  },
  lightbulb: {
    Icon: Lightbulb,
    containerClassName: 'w-2/3 h-4/5 bg-gradient-to-t from-[#111] to-[#222] skew-x-6 group-hover:skew-x-0 transition-transform duration-700 ease-out border border-border-muted flex items-center justify-center relative',
    overlayClassName: 'absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(245,165,36,0.15)_0%,transparent_60%)] opacity-0 group-hover:opacity-100 transition-opacity duration-500',
  },
  globe: {
    Icon: Globe,
    containerClassName: 'w-3/4 h-3/4 rounded-full bg-gradient-to-tr from-[#151515] to-[#2a2a2a] scale-90 group-hover:scale-100 transition-transform duration-700 ease-out border border-border-muted flex items-center justify-center relative shadow-[inset_0_0_20px_rgba(0,0,0,0.8)]',
    overlayClassName: 'absolute inset-0 rounded-full bg-[radial-gradient(circle_at_bottom_left,rgba(245,165,36,0.2)_0%,transparent_50%)] opacity-0 group-hover:opacity-100 transition-opacity duration-500',
  },
  car: {
    Icon: Car,
    containerClassName: 'w-4/5 h-2/3 bg-gradient-to-b from-[#222] to-[#111] -rotate-6 group-hover:rotate-0 transition-transform duration-700 ease-out border border-border-muted flex items-center justify-center relative',
    overlayClassName: 'absolute inset-0 bg-[linear-gradient(45deg,rgba(245,165,36,0.1)_0%,transparent_100%)] opacity-0 group-hover:opacity-100 transition-opacity duration-500',
  },
} as const;

export function ProjectCard({ project }: { project: ProjectRecord }) {
  const visual = projectVisuals[project.visual];
  const Icon = visual.Icon;
  const statusLabel = studioDefaults.jobStatusLabels[project.status];

  return (
    <Link href={`/studio?projectId=${project.id}`} className="group flex flex-col w-full bg-surface border border-border-muted hover:border-primary transition-colors cursor-pointer">
      <div className="w-full aspect-square relative overflow-hidden bg-background-dark border-b border-border-muted p-4 flex items-center justify-center">
        <div className={visual.containerClassName}>
          <div className={visual.overlayClassName}></div>
          <Icon className="w-10 h-10 text-text-muted group-hover:text-primary transition-colors duration-300 z-10" />
        </div>
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
  );
}

export function CreateProjectCard() {
  return (
    <Link href="/studio" className="group w-full aspect-square bg-surface/50 border border-dashed border-border-muted flex flex-col items-center justify-center gap-3 hover:border-primary hover:bg-surface-hover/50 transition-all duration-300">
      <div className="w-12 h-12 rounded-full bg-surface border border-border-muted flex items-center justify-center group-hover:bg-primary group-hover:border-primary group-hover:text-background-dark text-text-muted transition-all duration-300">
        <Plus className="w-6 h-6" />
      </div>
      <span className="text-sm font-display font-medium text-text-muted group-hover:text-primary transition-colors">{dashboardContent.createCardLabel}</span>
    </Link>
  );
}
