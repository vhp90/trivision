import type { ReactNode } from 'react';
import Link from 'next/link';
import { Box } from 'lucide-react';
import { brand, dashboardContent } from '@/content/site';
import type { ShellSummary } from '@/lib/db/types';

type WorkspaceShellProps = {
  summary: ShellSummary;
  activeSection: 'dashboard' | 'workspaces' | 'recent-generations' | 'favorites' | 'profile' | 'settings';
  title: string;
  description: string;
  actions?: ReactNode;
  children: ReactNode;
};

export function WorkspaceShell({
  summary,
  activeSection,
  title,
  description,
  actions,
  children,
}: WorkspaceShellProps) {
  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <header className="flex items-center justify-between whitespace-nowrap border-b border-border-muted bg-surface px-6 py-3 h-16 shrink-0">
        <div className="flex items-center gap-4">
          <Link href="/" prefetch={false} className="flex items-center gap-3 text-text-main">
            <div className="w-5 h-5 text-primary">
              <Box className="w-full h-full" />
            </div>
            <h2 className="text-xl font-display font-bold leading-tight tracking-tight">{brand.name}</h2>
          </Link>
          <div className="hidden lg:flex flex-col">
            <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">{summary.user.roleLabel}</span>
            <span className="text-[11px] font-mono text-primary">{summary.user.region} {'//'} {summary.user.engineVersion}</span>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <form action="/api/auth/logout" method="post">
            <button type="submit" className="text-[11px] font-mono uppercase tracking-[0.2em] text-text-muted hover:text-primary transition-colors">
              Logout
            </button>
          </form>
          <Link prefetch={false} href="/profile" className="flex items-center gap-3 hover:text-primary transition-colors">
            <div className="text-right hidden sm:block">
              <div className="text-sm font-display font-bold text-text-main">{summary.user.fullName}</div>
              <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted">{summary.user.roleLabel}</div>
            </div>
            <div className="h-8 w-8 bg-surface-hover border border-border-muted flex items-center justify-center text-sm font-display font-bold text-primary hover:border-primary transition-colors">
              {summary.user.initials}
            </div>
          </Link>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <aside className="w-[240px] flex-shrink-0 bg-surface border-r border-border-muted flex flex-col justify-between hidden md:flex">
          <div className="p-4 flex flex-col gap-6 overflow-y-auto">
            <div className="flex flex-col gap-1">
              <h3 className="text-xs font-mono text-text-muted uppercase tracking-wider mb-2 px-3">{dashboardContent.workspaceTitle}</h3>
              {dashboardContent.workspaceNav.map((item) => {
                const Icon = item.icon;
                const isActive = activeSection === item.id;

                return (
                  <Link
                    key={item.id}
                    href={item.href}
                    prefetch={false}
                    className={isActive
                      ? 'flex items-center gap-3 px-3 py-2 bg-surface-hover border-l-2 border-primary text-text-main transition-colors group'
                      : 'flex items-center gap-3 px-3 py-2 text-text-muted hover:bg-surface-hover hover:text-text-main border-l-2 border-transparent hover:border-border-muted transition-colors'}
                  >
                    <Icon className={isActive ? 'w-4 h-4 text-primary' : 'w-4 h-4'} />
                    <span className="text-sm font-body font-medium">{item.label}</span>
                  </Link>
                );
              })}
            </div>

            <div className="flex flex-col gap-1">
              <h3 className="text-xs font-mono text-text-muted uppercase tracking-wider mb-2 px-3">{dashboardContent.accountTitle}</h3>
              {dashboardContent.accountNav.map((item) => {
                const Icon = item.icon;
                const isActive = activeSection === item.id;

                return (
                  <Link
                    key={item.id}
                    href={item.href}
                    prefetch={false}
                    className={isActive
                      ? 'flex items-center gap-3 px-3 py-2 bg-surface-hover border-l-2 border-primary text-text-main transition-colors'
                      : 'flex items-center gap-3 px-3 py-2 text-text-muted hover:bg-surface-hover hover:text-text-main border-l-2 border-transparent hover:border-border-muted transition-colors'}
                  >
                    <Icon className={isActive ? 'w-4 h-4 text-primary' : 'w-4 h-4'} />
                    <span className="text-sm font-body font-medium">{item.label}</span>
                  </Link>
                );
              })}
            </div>
          </div>
        </aside>

        <main className="flex-1 flex flex-col overflow-hidden bg-background-dark relative">
          <div className="flex items-center justify-between p-6 pb-4">
            <div>
              <h1 className="text-2xl font-display font-bold text-text-main">{title}</h1>
              <p className="text-sm text-text-muted font-mono mt-1">{description}</p>
            </div>
            {actions ? <div className="flex items-center gap-3">{actions}</div> : null}
          </div>

          <div className="flex-1 overflow-y-auto p-6 pt-2">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
