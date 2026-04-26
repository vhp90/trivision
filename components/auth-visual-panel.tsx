export function AuthVisualPanel() {
  return (
    <div className="hidden md:flex flex-1 render-bg relative items-center justify-center overflow-hidden">
      <div className="absolute inset-0 opacity-20 [background-image:linear-gradient(to_right,#333336_1px,transparent_1px),linear-gradient(to_bottom,#333336_1px,transparent_1px)] [background-size:48px_48px]" />
      <div className="absolute right-[12%] top-[14%] h-28 w-28 border border-primary/25" />
      <div className="absolute bottom-[12%] left-[10%] h-36 w-36 border border-border-muted" />
      <div className="relative h-[min(58vw,620px)] w-[min(58vw,620px)]">
        <div className="absolute inset-8 border border-border-muted bg-background-dark/40 shadow-2xl shadow-black/50" />
        <div className="absolute left-0 top-[18%] w-52 border border-border-muted bg-surface/90 p-4">
          <div className="mb-3 h-2 w-20 bg-primary/70" />
          <div className="space-y-2">
            <div className="h-2 w-full bg-border-muted" />
            <div className="h-2 w-2/3 bg-border-muted" />
          </div>
        </div>
        <div className="absolute bottom-[16%] right-0 w-48 border border-border-muted bg-surface/90 p-4">
          <div className="grid grid-cols-3 gap-2">
            <div className="aspect-square bg-primary/70" />
            <div className="aspect-square bg-border-muted" />
            <div className="aspect-square bg-border-muted" />
          </div>
          <div className="mt-4 h-2 w-24 bg-text-muted/40" />
        </div>

        <div className="absolute inset-0 flex items-center justify-center">
          <div className="absolute h-[72%] w-[72%] rotate-45 border border-primary/20" />
          <div className="absolute h-[52%] w-[52%] border border-border-muted motion-safe:animate-[spin_34s_linear_infinite]" />
          <svg className="relative h-[72%] w-[72%] text-primary drop-shadow-[0_0_34px_rgba(245,166,36,0.28)]" fill="none" viewBox="0 0 420 420">
            <path d="M210 36 348 116v168L210 384 72 284V116L210 36Z" stroke="currentColor" strokeWidth="2" />
            <path d="M210 36v174m0 174V210M72 116l138 94 138-94M72 284l138-74 138 74" stroke="currentColor" strokeWidth="1.3" />
            <path d="M150 130h120l64 80-124 104L86 210l64-80Z" fill="currentColor" fillOpacity="0.12" stroke="currentColor" strokeWidth="2.2" />
            <path d="M150 130 210 314l60-184M86 210h248" stroke="currentColor" strokeWidth="1.3" />
            <circle cx="210" cy="210" r="6" fill="currentColor" />
            <circle cx="150" cy="130" r="4" fill="currentColor" />
            <circle cx="270" cy="130" r="4" fill="currentColor" />
            <circle cx="210" cy="314" r="4" fill="currentColor" />
          </svg>
        </div>
      </div>
    </div>
  );
}
