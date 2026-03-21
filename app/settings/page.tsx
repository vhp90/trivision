import { collectionPageContent } from '@/content/site';
import { requireAuthenticatedUser } from '@/lib/auth/session';
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
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {sections.map((section) => (
          <div key={section.id} className="bg-surface border border-border-muted p-5 flex flex-col gap-5">
            <div>
              <h2 className="text-xl font-display font-bold text-text-main">{section.title}</h2>
              <p className="text-sm text-text-muted mt-2 leading-6">{section.description}</p>
            </div>
            <div className="space-y-3">
              {section.items.map((item) => (
                <div key={item.id} className="bg-background-dark border border-border-muted p-3">
                  <div className="flex items-center justify-between gap-4">
                    <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">{item.label}</span>
                    <span className="text-[11px] font-mono text-primary">{item.value}</span>
                  </div>
                  <p className="text-sm text-text-main mt-3 leading-6">{item.description}</p>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </WorkspaceShell>
  );
}
