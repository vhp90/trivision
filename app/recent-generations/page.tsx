import Link from 'next/link';
import { Plus } from 'lucide-react';
import { collectionPageContent } from '@/content/site';
import { studioDefaults } from '@/lib/config/app';
import { requireAuthenticatedUser } from '@/lib/auth/session';
import { WorkspaceShell } from '@/components/workspace-shell';
import { getProjects, getShellSummary } from '@/lib/db/repository';

export const dynamic = 'force-dynamic';

export default async function RecentGenerationsPage() {
  const user = await requireAuthenticatedUser();
  const [summary, projects] = await Promise.all([
    getShellSummary(user.id),
    getProjects(user.id, { recentOnly: true }),
  ]);

  return (
    <WorkspaceShell
      summary={summary}
      activeSection="recent-generations"
      title={collectionPageContent.recent.title}
      description={collectionPageContent.recent.description}
      actions={
        <Link href="/studio" className="h-9 px-4 flex items-center justify-center gap-2 bg-primary text-background-dark text-sm font-display font-bold hover:bg-primary-hover transition-colors">
          <Plus className="w-4 h-4" />
          New Generation
        </Link>
      }
    >
      <div className="border border-border-muted bg-surface divide-y divide-border-muted">
        {projects.map((project) => (
          <Link
            key={project.id}
            href={`/studio?projectId=${project.id}`}
            className="grid grid-cols-[minmax(0,2fr)_120px_120px_160px] items-center gap-4 px-5 py-4 hover:bg-surface-hover transition-colors"
          >
            <div>
              <div className="text-sm font-body font-medium text-text-main">{project.name}</div>
              <div className="text-[11px] font-mono text-text-muted mt-1">
                {project.workspaceName} {'//'} {project.prompt || project.modelId || 'Provider-backed generation'}
              </div>
            </div>
            <div className="text-[11px] font-mono text-text-muted">{project.updatedLabel}</div>
            <div className="text-[11px] font-mono text-text-muted">{studioDefaults.jobStatusLabels[project.status]}</div>
            <div className="flex items-center justify-end gap-2">
              {project.format ? (
                <span className="text-[10px] font-mono text-primary border border-border-muted px-2 py-1">{project.format}</span>
              ) : null}
              <span className="text-[10px] font-mono text-text-main border border-primary/30 px-2 py-1">{project.outputFormat?.toUpperCase() ?? 'GLB'}</span>
            </div>
          </Link>
        ))}
      </div>
    </WorkspaceShell>
  );
}
