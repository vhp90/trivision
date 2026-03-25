import { collectionPageContent } from '@/content/site';
import { requireAuthenticatedUser } from '@/lib/auth/session';
import { SettingsPreferencesForm } from '@/components/settings-preferences-form';
import { WorkspaceShell } from '@/components/workspace-shell';
import { getSettings, getShellSummary } from '@/lib/db/repository';

export const dynamic = 'force-dynamic';

export default async function SettingsPage() {
  const user = await requireAuthenticatedUser();
  const [summary, sections] = await Promise.all([
    getShellSummary(user.id),
    getSettings(user.id),
  ]);

  return (
    <WorkspaceShell
      summary={summary}
      activeSection="settings"
      title={collectionPageContent.settings.title}
      description={collectionPageContent.settings.description}
    >
      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.7fr)_320px] gap-6 items-start">
        <SettingsPreferencesForm sections={sections} />

        <div className="bg-surface border border-border-muted p-6">
          <h2 className="text-xl font-display font-bold text-text-main">Runtime Snapshot</h2>
          <div className="mt-5 space-y-3">
            <div className="border border-border-muted bg-background-dark p-4">
              <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Region</div>
              <div className="mt-2 text-sm text-text-main">{summary.user.region}</div>
            </div>
            <div className="border border-border-muted bg-background-dark p-4">
              <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Session</div>
              <div className="mt-2 text-sm text-primary">{summary.user.sessionLabel}</div>
            </div>
            <div className="border border-border-muted bg-background-dark p-4">
              <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Engine</div>
              <div className="mt-2 text-sm text-text-main">{summary.user.engineVersion}</div>
            </div>
          </div>
        </div>
      </div>
    </WorkspaceShell>
  );
}
