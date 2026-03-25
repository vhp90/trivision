'use client';

import type { MouseEvent } from 'react';
import Link from 'next/link';
import { Rocket } from 'lucide-react';
import { motion, useMotionValue, useSpring, useTransform } from 'motion/react';
import { brand, landingPageContent } from '@/content/site';

export default function LandingPage() {
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    const { clientX, clientY } = e;
    const { innerWidth, innerHeight } = window;
    // Normalize mouse coordinates to range [-1, 1]
    const x = (clientX / innerWidth) * 2 - 1;
    const y = (clientY / innerHeight) * 2 - 1;
    mouseX.set(x);
    mouseY.set(y);
  };

  // Smooth spring physics for the 3D rotation
  const springConfig = { damping: 25, stiffness: 150 };
  const rotateX = useSpring(useTransform(mouseY, [-1, 1], [25, -25]), springConfig);
  const rotateY = useSpring(useTransform(mouseX, [-1, 1], [-25, 25]), springConfig);
  const translateZ = useSpring(useTransform(mouseY, [-1, 1], [-50, 50]), springConfig);

  return (
    <div 
      className="min-h-screen flex flex-col overflow-hidden relative bg-background-dark"
      onMouseMove={handleMouseMove}
    >
      <header className="h-16 border-b border-border-muted bg-background-dark/80 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          {/* Animated Trivision Logo */}
          <motion.div 
            animate={{ rotateY: 360 }} 
            transition={{ duration: 6, repeat: Infinity, ease: "linear" }}
            className="w-6 h-6 text-primary flex items-center justify-center"
            style={{ transformStyle: "preserve-3d" }}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-full h-full drop-shadow-[0_0_8px_rgba(245,165,36,0.5)]">
              <polygon points="12 2 2 7 12 12 22 7 12 2" fill="currentColor" fillOpacity="0.2" />
              <polyline points="2 17 12 22 22 17" />
              <polyline points="2 12 12 17 22 12" />
              <line x1="12" y1="22" x2="12" y2="12" />
            </svg>
          </motion.div>
          <h1 className="font-display font-bold text-lg tracking-tight text-text-main">{brand.name}</h1>
        </div>
        <nav className="flex items-center gap-6 font-mono text-xs">
          <Link prefetch={false} href="/login" className="text-text-muted hover:text-text-main transition-colors uppercase tracking-widest">Login</Link>
          <Link prefetch={false} href="/signup" className="text-primary hover:text-primary-hover transition-colors uppercase tracking-widest border border-primary/30 px-4 py-2 hover:border-primary">Sign Up</Link>
        </nav>
      </header>

      <main className="flex-1 relative flex items-center justify-center overflow-hidden" style={{ perspective: 1200 }}>
        <div className="absolute inset-0 flex items-center justify-center z-0 overflow-hidden pointer-events-none">
          <div className="absolute inset-0 hero-gradient z-10"></div>
          <div className="absolute inset-0 top-1/2 wireframe-bg z-0 animate-in fade-in duration-1000"></div>
          
          {/* Interactive 3D Geometry Pattern */}
          <motion.div 
            className="relative w-[600px] h-[600px] opacity-60 z-0 flex items-center justify-center"
            style={{ rotateX, rotateY, z: translateZ, transformStyle: "preserve-3d" }}
          >
            <motion.svg 
              animate={{ rotateZ: 360 }}
              transition={{ duration: 80, repeat: Infinity, ease: "linear" }}
              className="w-full h-full text-primary absolute" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="0.5" 
              viewBox="0 0 200 200"
              style={{ transformStyle: "preserve-3d", transform: "translateZ(50px)" }}
            >
              {/* Complex geometric shape */}
              <path d="M100 10 L190 60 L190 140 L100 190 L10 140 L10 60 Z"></path>
              <path d="M100 10 L150 100 L190 60"></path>
              <path d="M100 10 L50 100 L10 60"></path>
              <path d="M100 190 L150 100 L190 140"></path>
              <path d="M100 190 L50 100 L10 140"></path>
              <path d="M10 60 L50 100 L10 140"></path>
              <path d="M190 60 L150 100 L190 140"></path>
              <path d="M100 10 L100 190"></path>
              <path d="M10 60 L190 140"></path>
              <path d="M190 60 L10 140"></path>
              <circle cx="100" cy="100" r="40" strokeDasharray="2 2" strokeWidth="0.2"></circle>
              <path d="M100 60 L135 120 L65 120 Z" strokeWidth="0.8"></path>
              <path d="M100 140 L135 80 L65 80 Z" strokeWidth="0.8"></path>
            </motion.svg>

            {/* Inner floating elements for parallax depth */}
            <motion.div 
              className="absolute w-40 h-40 border border-primary/40 rounded-full"
              style={{ transform: "translateZ(120px)" }}
              animate={{ scale: [1, 1.1, 1], opacity: [0.4, 0.8, 0.4] }}
              transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
            />
            <motion.div 
              className="absolute w-20 h-20 border border-primary/60 rotate-45"
              style={{ transform: "translateZ(180px)" }}
              animate={{ rotateZ: [45, 225] }}
              transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
            />
          </motion.div>
        </div>

        <div className="relative z-20 flex flex-col items-center text-center max-w-4xl px-6 animate-in fade-in slide-in-from-bottom-4 duration-1000 delay-300 fill-mode-both pointer-events-none">
          <div className="mb-8 inline-flex items-center gap-2 px-3 py-1 border border-border-muted bg-surface/50 font-mono text-[10px] text-text-muted uppercase tracking-widest backdrop-blur-sm">
            <span className="w-1.5 h-1.5 bg-success rounded-full animate-pulse"></span>
            {landingPageContent.statusLabel}
          </div>
          
          <h2 className="font-display font-bold text-[56px] md:text-[64px] leading-[1.1] tracking-tight text-text-main mb-6 drop-shadow-lg">
            {landingPageContent.heading.primary}<br/>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-text-main to-text-muted">{landingPageContent.heading.secondary}</span>
          </h2>
          
          <p className="font-body text-text-muted text-lg md:text-xl max-w-2xl mb-12 font-light">
            {landingPageContent.description}
          </p>
          
          <div className="pointer-events-auto">
            <Link prefetch={false} href={landingPageContent.primaryCta.href} className="h-12 px-8 bg-primary hover:bg-primary-hover hover:shadow-[0_0_15px_rgba(245,166,35,0.3)] transition-all text-background-dark font-display font-medium text-[15px] uppercase tracking-wider flex items-center justify-center gap-3 w-full max-w-[280px]">
              <Rocket className="w-5 h-5" />
              {landingPageContent.primaryCta.label}
            </Link>
          </div>
          
          <div className="mt-16 font-mono text-[11px] text-text-muted/50 text-left w-full max-w-lg p-4 border border-border-muted/30 bg-surface/20 backdrop-blur-sm">
            {landingPageContent.terminalLines.map((line) => (
              <div key={line}>{line}</div>
            ))}
          </div>
        </div>
      </main>

      <footer className="absolute bottom-0 w-full p-4 flex justify-between items-center font-mono text-[10px] text-text-muted/40 z-20 pointer-events-none">
        <div>{landingPageContent.footer.left}</div>
        <div>{landingPageContent.footer.right}</div>
      </footer>
    </div>
  );
}
