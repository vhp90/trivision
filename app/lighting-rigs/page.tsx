import { collectionPageContent } from '@/content/site';
import { requireAuthenticatedUser } from '@/lib/auth/session';
import { WorkspaceShell } from '@/components/workspace-shell';
import { getLightingRigs, getShellSummary } from '@/lib/db/repository';

export const dynamic = 'force-dynamic';

export default async function LightingRigsPage() {
  const user = await requireAuthenticatedUser();
  const [summary, lightingRigs] = await Promise.all([
    getShellSummary(user.id),
    getLightingRigs(user.id),
  ]);

  return (
    <WorkspaceShell
      summary={summary}
      activeSection="lighting-rigs"
      title={collectionPageContent.lightingRigs.title}
      description={collectionPageContent.lightingRigs.description}
    >
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {lightingRigs.map((rig) => (
          <div key={rig.id} className="bg-surface border border-border-muted p-5 flex flex-col gap-4 hover:border-primary transition-colors">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="text-[10px] font-mono uppercase tracking-[0.3em] text-text-muted">{rig.rigType}</div>
                <h2 className="text-xl font-display font-bold text-text-main mt-2">{rig.name}</h2>
              </div>
              <span className="text-[10px] font-mono border border-primary/30 px-2 py-1 text-primary uppercase tracking-[0.2em]">{rig.temperature}</span>
            </div>
            <div className="border-t border-border-muted pt-4 space-y-2">
              <div className="flex items-center justify-between text-[11px] font-mono">
                <span className="text-text-muted">Mood</span>
                <span className="text-text-main">{rig.mood}</span>
              </div>
              <div className="flex items-center justify-between text-[11px] font-mono">
                <span className="text-text-muted">Usage</span>
                <span className="text-text-main">{rig.usageLabel}</span>
              </div>
            </div>
            <div className="mt-auto text-[11px] font-mono text-text-muted">{rig.updatedLabel}</div>
          </div>
        ))}
      </div>
    </WorkspaceShell>
  );
}
