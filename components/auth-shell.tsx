import type { ReactNode } from 'react';
import Link from 'next/link';
import { Box } from 'lucide-react';
import { authShellContent, brand } from '@/content/site';
import { AuthVisualPanel } from '@/components/auth-visual-panel';

type AuthShellProps = {
  title: string;
  subtitle: string;
  children: ReactNode;
};

export function AuthShell({ title, subtitle, children }: AuthShellProps) {
  return (
    <div className="h-screen w-full overflow-hidden flex">
      <div className="w-full md:w-[30%] min-w-[320px] md:min-w-[400px] max-w-[500px] h-full bg-surface flex flex-col border-r border-border-muted relative z-10 flex-shrink-0">
        <div className="h-16 flex items-center px-8 border-b border-border-muted">
          <Link href="/" className="flex items-center gap-2 text-primary">
            <Box className="w-6 h-6" />
            <span className="font-display font-bold tracking-tight text-lg text-text-main">{brand.uppercaseName}</span>
          </Link>
        </div>

        <div className="flex-1 flex flex-col justify-center px-8 sm:px-12 py-8 overflow-y-auto">
          <div className="mb-10">
            <h1 className="font-display font-bold text-3xl text-text-main mb-2">{title}</h1>
            <p className="text-text-muted text-sm font-body">{subtitle}</p>
          </div>

          {children}

          <div className="mt-auto pt-12">
            <p className="text-[10px] font-mono text-border-muted">{authShellContent.footerLabel}</p>
          </div>
        </div>
      </div>

      <AuthVisualPanel />
    </div>
  );
}
