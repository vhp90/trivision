import Link from 'next/link';
import { Settings } from 'lucide-react';
import { collectionPageContent } from '@/content/site';
import { requireAuthenticatedUser } from '@/lib/auth/session';
import { ProfileForm } from '@/components/profile-form';
import { WorkspaceShell } from '@/components/workspace-shell';
import { getShellSummary, getWorkspaces } from '@/lib/db/repository';

export const dynamic = 'force-dynamic';

export default async function ProfilePage() {
  const user = await requireAuthenticatedUser();
  const [summary, workspaces] = await Promise.all([
    getShellSummary(user.id),
    getWorkspaces(user.id),
  ]);
  const primaryWorkspace = workspaces.find((workspace) => workspace.isPrimary) ?? workspaces[0] ?? null;

  return (
    <WorkspaceShell
      summary={summary}
      activeSection="profile"
      title={collectionPageContent.profile.title}
      description={collectionPageContent.profile.description}
      actions={
        <Link prefetch={false} href="/settings" className="h-9 px-4 flex items-center justify-center gap-2 border border-border-muted bg-surface text-text-main text-sm font-body hover:bg-surface-hover hover:border-primary/50 transition-colors">
          <Settings className="w-4 h-4" />
          Workspace Settings
        </Link>
      }
    >
      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.7fr)_360px] gap-6 items-start">
        <ProfileForm user={summary.user} />

        <div className="space-y-6">
          <div className="bg-surface border border-border-muted p-6">
            <h2 className="text-xl font-display font-bold text-text-main">Account Snapshot</h2>
            <div className="mt-5 space-y-3">
              <div className="border border-border-muted bg-background-dark p-4">
                <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Workspace</div>
                <div className="mt-2 text-sm text-text-main">{primaryWorkspace?.name ?? 'No workspace yet'}</div>
              </div>
              <div className="border border-border-muted bg-background-dark p-4">
                <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Role</div>
                <div className="mt-2 text-sm text-text-main">{summary.user.roleLabel}</div>
              </div>
              <div className="border border-border-muted bg-background-dark p-4">
                <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Latency</div>
                <div className="mt-2 text-sm text-primary">{summary.user.latencyLabel}</div>
              </div>
            </div>
          </div>

          <div className="bg-surface border border-border-muted p-6">
            <h2 className="text-xl font-display font-bold text-text-main">Workspace Coverage</h2>
            <div className="mt-5 grid grid-cols-2 gap-4">
              <div className="border border-border-muted bg-background-dark p-4">
                <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Workspaces</div>
                <div className="mt-2 text-2xl font-display font-bold text-text-main">{workspaces.length}</div>
              </div>
              <div className="border border-border-muted bg-background-dark p-4">
                <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Engine</div>
                <div className="mt-2 text-sm font-mono text-primary">{summary.user.engineVersion}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </WorkspaceShell>
  );
}
