import Link from 'next/link';
import {
  ArrowRight,
  Box,
  Check,
  Download,
  Image as ImageIcon,
  Layers3,
  RotateCcw,
  Sparkles,
  Star,
  Wand2,
} from 'lucide-react';
import { BrandMark } from '@/components/brand-mark';
import { landingPageContent } from '@/content/site';

function StudioPreview() {
  return (
    <div className="relative mx-auto w-full max-w-[680px]">
      <div className="absolute -inset-8 bg-primary/10 blur-3xl" />
      <div className="relative border border-border-muted bg-surface shadow-2xl shadow-black/40">
        <div className="flex h-11 items-center justify-between border-b border-border-muted px-4">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 bg-primary" />
            <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-text-muted">Studio View</span>
          </div>
          <div className="flex items-center gap-2 text-text-muted">
            <span className="h-1.5 w-1.5 bg-border-muted" />
            <span className="h-1.5 w-1.5 bg-border-muted" />
            <span className="h-1.5 w-1.5 bg-border-muted" />
          </div>
        </div>

        <div className="grid min-h-[440px] grid-cols-[170px_1fr] md:grid-cols-[190px_1fr_150px]">
          <div className="border-r border-border-muted bg-background-dark/70 p-4">
            <div className="mb-5 flex items-center gap-2 text-text-main">
              <ImageIcon className="h-4 w-4 text-primary" />
              <span className="font-mono text-[10px] uppercase tracking-[0.2em]">Input</span>
            </div>
            <div className="aspect-square border border-border-muted bg-surface-hover">
              <div className="h-full w-full bg-[linear-gradient(135deg,rgba(245,166,36,0.26),transparent_42%),radial-gradient(circle_at_52%_42%,rgba(237,237,237,0.18),transparent_25%)]" />
            </div>
            <div className="mt-4 space-y-2">
              {['Mask ready', 'GLB output', '1024 texture'].map((item) => (
                <div key={item} className="flex items-center gap-2 border border-border-muted px-2 py-2 text-[10px] text-text-muted">
                  <Check className="h-3 w-3 text-primary" />
                  {item}
                </div>
              ))}
            </div>
          </div>

          <div className="relative flex items-center justify-center overflow-hidden bg-background-dark">
            <div className="absolute inset-0 opacity-25 [background-image:linear-gradient(to_right,#333336_1px,transparent_1px),linear-gradient(to_bottom,#333336_1px,transparent_1px)] [background-size:42px_42px]" />
            <div className="absolute h-[360px] w-[360px] border border-primary/20 motion-safe:animate-[spin_28s_linear_infinite]" />
            <div className="absolute h-[250px] w-[250px] rotate-45 border border-border-muted" />
            <svg className="relative h-[300px] w-[300px] text-primary drop-shadow-[0_0_30px_rgba(245,166,36,0.28)]" fill="none" viewBox="0 0 320 320">
              <path d="M160 24 270 88v128l-110 80-110-80V88L160 24Z" stroke="currentColor" strokeWidth="2" />
              <path d="M160 24v136m0 136V160M50 88l110 72 110-72M50 216l110-56 110 56" stroke="currentColor" strokeWidth="1.2" />
              <path d="M112 98h96l48 62-96 80-96-80 48-62Z" fill="currentColor" fillOpacity="0.12" stroke="currentColor" strokeWidth="2" />
              <path d="M112 98 160 240l48-142M64 160h192" stroke="currentColor" strokeWidth="1.2" />
              <circle cx="160" cy="160" r="5" fill="currentColor" />
            </svg>
          </div>

          <div className="hidden border-l border-border-muted bg-background-dark/70 p-4 md:block">
            <div className="mb-5 flex items-center gap-2 text-text-main">
              <Layers3 className="h-4 w-4 text-primary" />
              <span className="font-mono text-[10px] uppercase tracking-[0.2em]">Asset</span>
            </div>
            <div className="space-y-3">
              {[
                ['Format', 'GLB'],
                ['Status', 'Ready'],
                ['View', 'Solid'],
                ['Retry', 'Available'],
              ].map(([label, value]) => (
                <div key={label} className="border-b border-border-muted pb-2">
                  <div className="font-mono text-[9px] uppercase tracking-[0.2em] text-text-muted">{label}</div>
                  <div className="mt-1 text-sm font-semibold text-text-main">{value}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function SectionLabel({ children }: { children: string }) {
  return (
    <div className="mb-4 font-mono text-[10px] uppercase tracking-[0.28em] text-primary">
      {children}
    </div>
  );
}

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background-dark text-text-main">
      <header className="sticky top-0 z-50 border-b border-border-muted bg-background-dark/88 px-5 backdrop-blur-md">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between">
          <Link href="/" prefetch={false} aria-label="Trivision home">
            <BrandMark size="md" />
          </Link>
          <nav className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.2em]">
            <Link href="#workflow" className="hidden px-3 py-2 text-text-muted transition-colors hover:text-text-main md:inline-flex">
              Workflow
            </Link>
            <Link href="#studio" className="hidden px-3 py-2 text-text-muted transition-colors hover:text-text-main md:inline-flex">
              Studio
            </Link>
            <Link href="/login" prefetch={false} className="px-3 py-2 text-text-muted transition-colors hover:text-text-main">
              Login
            </Link>
            <Link href="/signup" prefetch={false} className="border border-primary/50 bg-primary px-4 py-2 font-bold text-background-dark transition-colors hover:bg-primary-hover">
              Sign Up
            </Link>
          </nav>
        </div>
      </header>

      <main>
        <section className="relative overflow-hidden border-b border-border-muted">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_72%_30%,rgba(245,166,36,0.14),transparent_30%),linear-gradient(180deg,rgba(24,24,26,0.78),rgba(9,9,10,1))]" />
          <div className="absolute inset-x-0 bottom-0 h-1/2 opacity-20 [background-image:linear-gradient(to_right,#333336_1px,transparent_1px),linear-gradient(to_bottom,#333336_1px,transparent_1px)] [background-size:56px_56px]" />
          <div className="relative mx-auto grid min-h-[calc(100svh-4rem)] max-w-7xl items-center gap-12 px-5 py-16 lg:grid-cols-[0.9fr_1.1fr] lg:py-20">
            <div>
              <div className="mb-8 inline-flex border border-border-muted bg-surface/70 px-3 py-2 font-mono text-[10px] uppercase tracking-[0.22em] text-text-muted">
                {landingPageContent.statusLabel}
              </div>
              <h1 className="font-display text-6xl font-bold leading-none tracking-tight text-text-main md:text-7xl lg:text-8xl">
                {landingPageContent.heading.primary}
              </h1>
              <p className="mt-6 max-w-2xl font-display text-3xl font-semibold leading-tight text-text-main md:text-4xl">
                {landingPageContent.heading.secondary}
              </p>
              <p className="mt-6 max-w-xl text-base leading-7 text-text-muted md:text-lg">
                {landingPageContent.description}
              </p>
              <div className="mt-9 flex flex-col gap-3 sm:flex-row">
                <Link href={landingPageContent.primaryCta.href} prefetch={false} className="inline-flex h-12 items-center justify-center gap-3 bg-primary px-6 font-display text-sm font-bold uppercase tracking-[0.16em] text-background-dark transition-colors hover:bg-primary-hover">
                  {landingPageContent.primaryCta.label}
                  <ArrowRight className="h-4 w-4" />
                </Link>
                <Link href={landingPageContent.secondaryCta.href} prefetch={false} className="inline-flex h-12 items-center justify-center border border-border-muted px-6 font-display text-sm font-bold uppercase tracking-[0.16em] text-text-main transition-colors hover:border-primary hover:text-primary">
                  {landingPageContent.secondaryCta.label}
                </Link>
              </div>
            </div>

            <StudioPreview />
          </div>
        </section>

        <section id="workflow" className="border-b border-border-muted bg-surface">
          <div className="mx-auto grid max-w-7xl grid-cols-1 md:grid-cols-4">
            {landingPageContent.workflow.map((item, index) => (
              <div key={item.title} className="border-b border-border-muted p-6 md:border-b-0 md:border-r last:md:border-r-0">
                <div className="font-mono text-[10px] uppercase tracking-[0.24em] text-primary">
                  0{index + 1}
                </div>
                <h2 className="mt-4 font-display text-2xl font-bold text-text-main">{item.title}</h2>
                <p className="mt-3 text-sm leading-6 text-text-muted">{item.description}</p>
              </div>
            ))}
          </div>
        </section>

        <section id="studio" className="border-b border-border-muted">
          <div className="mx-auto grid max-w-7xl gap-10 px-5 py-20 lg:grid-cols-[0.8fr_1.2fr]">
            <div>
              <SectionLabel>Workspace</SectionLabel>
              <h2 className="max-w-xl font-display text-4xl font-bold leading-tight text-text-main md:text-5xl">
                Everything needed to produce and manage generated 3D assets.
              </h2>
            </div>
            <div className="grid gap-4 md:grid-cols-3">
              {landingPageContent.capabilities.map((item) => (
                <article key={item.title} className="border border-border-muted bg-surface p-5">
                  <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-primary">{item.eyebrow}</div>
                  <h3 className="mt-4 font-display text-xl font-bold leading-tight text-text-main">{item.title}</h3>
                  <p className="mt-4 text-sm leading-6 text-text-muted">{item.description}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="border-b border-border-muted bg-surface/55">
          <div className="mx-auto grid max-w-7xl gap-10 px-5 py-20 lg:grid-cols-2">
            <div className="border border-border-muted bg-background-dark p-6">
              <div className="flex items-center justify-between border-b border-border-muted pb-4">
                <div>
                  <SectionLabel>Model Surface</SectionLabel>
                  <h2 className="font-display text-3xl font-bold text-text-main">Choose the right generation path.</h2>
                </div>
                <Sparkles className="h-6 w-6 text-primary" />
              </div>
              <div className="mt-6 space-y-3">
                {landingPageContent.modelSurfaces.map((model) => (
                  <div key={model} className="flex items-center justify-between border border-border-muted bg-surface px-4 py-3">
                    <div className="font-display font-semibold text-text-main">{model}</div>
                    <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-text-muted">Available</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              {[
                { icon: Wand2, title: 'Preprocess', text: 'Prepare reference images before generation.' },
                { icon: RotateCcw, title: 'Retry', text: 'Run another attempt from a saved generation.' },
                { icon: Star, title: 'Favorite', text: 'Pin useful outputs for quick review.' },
                { icon: Download, title: 'Export', text: 'Download generated assets from the studio.' },
              ].map((item) => {
                const Icon = item.icon;
                return (
                  <article key={item.title} className="border border-border-muted bg-background-dark p-5">
                    <Icon className="h-5 w-5 text-primary" />
                    <h3 className="mt-5 font-display text-xl font-bold text-text-main">{item.title}</h3>
                    <p className="mt-3 text-sm leading-6 text-text-muted">{item.text}</p>
                  </article>
                );
              })}
            </div>
          </div>
        </section>

        <section className="px-5 py-20">
          <div className="mx-auto flex max-w-7xl flex-col items-start justify-between gap-8 border-y border-border-muted py-12 md:flex-row md:items-center">
            <div>
              <SectionLabel>Trivision Studio</SectionLabel>
              <h2 className="font-display text-4xl font-bold text-text-main md:text-5xl">
                Start with a reference. Leave with an asset.
              </h2>
            </div>
            <Link href="/signup" prefetch={false} className="inline-flex h-12 items-center justify-center gap-3 bg-primary px-6 font-display text-sm font-bold uppercase tracking-[0.16em] text-background-dark transition-colors hover:bg-primary-hover">
              Create Workspace
              <Box className="h-4 w-4" />
            </Link>
          </div>
        </section>
      </main>

      <footer className="border-t border-border-muted bg-surface px-5 py-6">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <BrandMark size="sm" />
          <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-text-muted">
            Image to 3D asset generation workspace
          </div>
        </div>
      </footer>
    </div>
  );
}
