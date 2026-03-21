import { collectionPageContent } from '@/content/site';
import { requireAuthenticatedUser } from '@/lib/auth/session';
import { WorkspaceShell } from '@/components/workspace-shell';
import { getMaterials, getShellSummary } from '@/lib/db/repository';

export const dynamic = 'force-dynamic';

export default async function MaterialsPage() {
  const user = await requireAuthenticatedUser();
  const [summary, materials] = await Promise.all([
    getShellSummary(user.id),
    getMaterials(user.id),
  ]);

  return (
    <WorkspaceShell
      summary={summary}
      activeSection="materials"
      title={collectionPageContent.materials.title}
      description={collectionPageContent.materials.description}
    >
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {materials.map((material) => (
          <div key={material.id} className="bg-surface border border-border-muted p-5 hover:border-primary transition-colors">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="text-[10px] font-mono uppercase tracking-[0.3em] text-text-muted">{material.category}</div>
                <h2 className="text-xl font-display font-bold text-text-main mt-2">{material.name}</h2>
              </div>
              <div className="text-[10px] font-mono border border-primary/30 px-2 py-1 text-primary uppercase tracking-[0.2em]">{material.finish}</div>
            </div>
            <div className="grid grid-cols-2 gap-3 mt-5">
              <div className="bg-background-dark border border-border-muted p-3">
                <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Palette</div>
                <div className="text-sm text-text-main mt-2">{material.palette}</div>
              </div>
              <div className="bg-background-dark border border-border-muted p-3">
                <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">Usage</div>
                <div className="text-sm text-text-main mt-2">{material.usageLabel}</div>
              </div>
            </div>
            <div className="mt-4 text-[11px] font-mono text-text-muted">{material.updatedLabel}</div>
          </div>
        ))}
      </div>
    </WorkspaceShell>
  );
}
