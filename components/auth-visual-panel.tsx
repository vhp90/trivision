import { authVisualContent } from '@/content/site';

export function AuthVisualPanel() {
  return (
    <div className="hidden md:flex flex-1 render-bg relative flex-col justify-between items-end p-8">
      <div className="flex items-center gap-2 bg-surface/80 backdrop-blur-sm border border-border-muted px-3 py-1.5">
        <div className="w-2 h-2 rounded-full bg-success shadow-[0_0_8px_#00E676]"></div>
        <span className="text-[11px] font-mono text-text-muted uppercase">{authVisualContent.panelStatus}</span>
      </div>

      <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-80 mix-blend-screen">
        <svg className="text-primary/20" fill="none" height="600" viewBox="0 0 600 600" width="600" xmlns="http://www.w3.org/2000/svg">
          <path d="M300 100 L500 250 L300 400 L100 250 Z" stroke="currentColor" strokeDasharray="4 4" strokeWidth="1"></path>
          <path d="M300 200 L400 275 L300 350 L200 275 Z" stroke="currentColor" strokeWidth="2"></path>
          <path d="M300 100 L300 400" stroke="currentColor" strokeWidth="1"></path>
          <path d="M100 250 L500 250" stroke="currentColor" strokeWidth="1"></path>
          <circle cx="300" cy="275" fill="#F5A623" r="4"></circle>
          <circle cx="400" cy="275" fill="currentColor" r="3"></circle>
          <circle cx="200" cy="275" fill="currentColor" r="3"></circle>
          <circle cx="300" cy="100" fill="currentColor" r="3"></circle>
          <circle cx="300" cy="400" fill="currentColor" r="3"></circle>
        </svg>
      </div>

      <div className="absolute left-8 bottom-8 text-[11px] font-mono text-text-muted flex flex-col gap-1 opacity-50">
        {authVisualContent.panelDiagnostics.map((item) => (
          <p key={item.label}>
            {item.label}{' '}
            <span className={item.label === 'DATA_STREAM_ACTIVE //' ? 'text-primary' : undefined}>{item.value}</span>
          </p>
        ))}
      </div>
    </div>
  );
}
