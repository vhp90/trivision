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
  const visibleSettingIds = new Set([
    'default-model',
    'default-output-format',
  ]);
  const visibleSections = sections
    .map((section) => ({
      ...section,
      items: section.items.filter((item) => visibleSettingIds.has(item.key ?? item.id)),
    }))
    .filter((section) => section.items.length > 0);
  const optionCount = visibleSections.reduce((total, section) => total + section.items.length, 0);
  const defaultModel = visibleSections
    .flatMap((section) => section.items)
    .find((item) => (item.key ?? item.id) === 'default-model')?.value ?? 'Configured';

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
          <h2 className="text-xl font-display font-bold text-text-main">Preference Summary</h2>
          <div className="mt-5 space-y-3">
            <div className="border border-border-muted bg-background-dark p-4">
              <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Sections</div>
              <div className="mt-2 text-2xl font-display font-bold text-text-main">{visibleSections.length}</div>
            </div>
            <div className="border border-border-muted bg-background-dark p-4">
              <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Options</div>
              <div className="mt-2 text-2xl font-display font-bold text-text-main">{optionCount}</div>
            </div>
            <div className="border border-border-muted bg-background-dark p-4">
              <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Default Model</div>
              <div className="mt-2 text-sm text-primary">{defaultModel}</div>
            </div>
            <div className="border border-border-muted bg-background-dark p-4">
              <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Profile</div>
              <div className="mt-2 text-sm text-text-main">{summary.user.fullName}</div>
            </div>
          </div>
        </div>
      </div>
    </WorkspaceShell>
  );
}
