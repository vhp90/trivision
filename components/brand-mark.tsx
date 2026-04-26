type BrandMarkProps = {
  size?: 'sm' | 'md' | 'lg';
  showWordmark?: boolean;
  className?: string;
};

const sizeClasses = {
  sm: 'h-7',
  md: 'h-9',
  lg: 'h-12',
} as const;

const iconSizeClasses = {
  sm: 'h-7 w-7',
  md: 'h-9 w-9',
  lg: 'h-12 w-12',
} as const;

export function BrandMark({ size = 'md', showWordmark = true, className = '' }: BrandMarkProps) {
  return (
    <div className={`flex items-center gap-3 ${sizeClasses[size]} ${className}`}>
      <div className={`${iconSizeClasses[size]} relative flex items-center justify-center border border-primary/45 bg-primary/10 text-primary shadow-[0_0_24px_rgba(245,166,36,0.16)]`}>
        <svg aria-hidden="true" className="h-[70%] w-[70%]" fill="none" viewBox="0 0 40 40">
          <path d="M20 4 34 12v16L20 36 6 28V12L20 4Z" stroke="currentColor" strokeWidth="2" />
          <path d="M20 4v16m0 16V20M6 12l14 8 14-8M6 28l14-8 14 8" stroke="currentColor" strokeWidth="1.4" />
          <path d="M14 14h12l5 6-11 8-11-8 5-6Z" fill="currentColor" fillOpacity="0.2" stroke="currentColor" strokeWidth="1.4" />
        </svg>
      </div>
      {showWordmark ? (
        <div className="leading-none">
          <div className="font-display text-lg font-bold tracking-tight text-text-main">Trivision</div>
          <div className="mt-1 hidden font-mono text-[9px] uppercase tracking-[0.24em] text-text-muted sm:block">
            3D Asset Studio
          </div>
        </div>
      ) : null}
    </div>
  );
}
