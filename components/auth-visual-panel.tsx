import { authVisualContent } from '@/content/site';

export function AuthVisualPanel() {
  return (
    <div className="hidden md:flex flex-1 bg-background-dark relative flex-col justify-between p-8">
      <div className="flex items-center gap-2 bg-surface/80 backdrop-blur-sm border border-border-muted px-3 py-1.5">
        <div className="w-2 h-2 rounded-full bg-success shadow-[0_0_8px_var(--color-success)]"></div>
        <span className="text-[11px] font-mono text-text-muted uppercase">{authVisualContent.panelStatus}</span>
      </div>

      <div className="max-w-xl self-end border-l border-border-muted pl-6">
        <h2 className="font-display text-3xl font-bold leading-tight text-text-main">
          {authVisualContent.panelTitle}
        </h2>
        <div className="mt-6 flex flex-col gap-3">
          {authVisualContent.panelItems.map((item) => (
            <p key={item} className="border-b border-border-muted pb-3 text-sm text-text-muted">
              {item}
            </p>
          ))}
        </div>
      </div>
    </div>
  );
}
