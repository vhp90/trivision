import Link from 'next/link';
import { FolderOpen, Plus } from 'lucide-react';
import { collectionPageContent } from '@/content/site';
import { requireAuthenticatedUser } from '@/lib/auth/session';
import { WorkspaceShell } from '@/components/workspace-shell';
import { getShellSummary, getWorkspaces } from '@/lib/db/repository';

export const dynamic = 'force-dynamic';

export default async function WorkspacesPage() {
  const user = await requireAuthenticatedUser();
  const [summary, workspaces] = await Promise.all([
    getShellSummary(user.id),
    getWorkspaces(user.id),
  ]);

  return (
    <WorkspaceShell
      summary={summary}
      activeSection="workspaces"
      title={collectionPageContent.workspaces.title}
      description={collectionPageContent.workspaces.description}
      actions={
        <Link prefetch={false} href="/studio" className="h-9 px-4 flex items-center justify-center gap-2 bg-primary text-background-dark text-sm font-display font-bold hover:bg-primary-hover transition-colors">
          <Plus className="w-4 h-4" />
          New Generation
        </Link>
      }
    >
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        {workspaces.map((workspace) => (
          <div key={workspace.id} className="flex min-w-0 flex-col gap-5 border border-border-muted bg-surface p-5 transition-colors hover:border-primary">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <div className="text-[10px] font-mono uppercase tracking-[0.3em] text-text-muted">{workspace.code}</div>
                <h2 className="mt-2 truncate text-xl font-display font-bold text-text-main">{workspace.name}</h2>
              </div>
              <span className="shrink-0 border border-primary/30 px-2 py-1 text-[10px] font-mono uppercase tracking-[0.2em] text-primary">{workspace.status}</span>
            </div>
            <p className="text-sm text-text-muted leading-6">{workspace.description}</p>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-background-dark border border-border-muted p-3">
                <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Projects</div>
                <div className="text-2xl font-display font-bold text-text-main mt-2">{workspace.projectCount}</div>
              </div>
              <div className="bg-background-dark border border-border-muted p-3">
                <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Favorites</div>
                <div className="text-2xl font-display font-bold text-text-main mt-2">{workspace.favoriteCount}</div>
              </div>
            </div>
            <div className="border-t border-border-muted pt-4 space-y-2">
              <div className="flex items-center justify-between gap-4 text-[11px] font-mono">
                <span className="text-text-muted">Primary Focus</span>
                <span className="text-right text-text-main">{workspace.primaryFocus}</span>
              </div>
              <div className="flex items-center justify-between gap-4 text-[11px] font-mono">
                <span className="text-text-muted">Secondary Focus</span>
                <span className="text-right text-text-main">{workspace.secondaryFocus}</span>
              </div>
              <div className="flex items-center justify-between gap-4 text-[11px] font-mono">
                <span className="text-text-muted">Last Activity</span>
                <span className="text-right text-primary">{workspace.updatedLabel}</span>
              </div>
            </div>
            <Link prefetch={false} href="/dashboard" className="mt-auto h-10 border border-border-muted bg-background-dark hover:border-primary hover:text-primary transition-colors text-sm font-body text-text-main flex items-center justify-center gap-2">
              <FolderOpen className="w-4 h-4" />
              View Projects
            </Link>
          </div>
        ))}
      </div>
    </WorkspaceShell>
  );
}
