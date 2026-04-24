export function AuthVisualPanel() {
  return (
    <div className="hidden md:flex flex-1 render-bg relative items-center justify-center overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_65%_45%,rgba(245,166,35,0.16),transparent_32%),linear-gradient(135deg,rgba(255,255,255,0.04),transparent_35%)]" />
      <div className="relative h-[min(54vw,560px)] w-[min(54vw,560px)] opacity-85">
        <svg className="h-full w-full text-primary/35" fill="none" viewBox="0 0 600 600" xmlns="http://www.w3.org/2000/svg">
          <path d="M300 82 496 196v208L300 518 104 404V196L300 82Z" stroke="currentColor" strokeWidth="1.5" />
          <path d="M300 82v208m0 228V290M104 196l196 94 196-94M104 404l196-114 196 114" stroke="currentColor" strokeWidth="1" />
          <path d="M216 190h168l84 100-168 120-168-120 84-100Z" stroke="currentColor" strokeWidth="2" />
          <path d="M216 190 300 410l84-220M132 290h336" stroke="currentColor" strokeWidth="1" />
          <circle className="fill-primary" cx="300" cy="290" r="5" />
          <circle cx="216" cy="190" fill="currentColor" r="3" />
          <circle cx="384" cy="190" fill="currentColor" r="3" />
          <circle cx="300" cy="410" fill="currentColor" r="3" />
        </svg>
      </div>
    </div>
  );
}
