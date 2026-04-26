import type { ReactNode } from 'react';
import Link from 'next/link';
import { dashboardContent } from '@/content/site';
import { BrandMark } from '@/components/brand-mark';
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
      <header className="flex h-16 shrink-0 items-center justify-between gap-4 border-b border-border-muted bg-surface px-5 py-3">
        <div className="flex min-w-0 items-center gap-5">
          <Link href="/" prefetch={false} aria-label="Trivision home" className="shrink-0">
            <BrandMark size="sm" />
          </Link>
          <div className="hidden min-w-0 border-l border-border-muted pl-5 lg:flex lg:flex-col">
            <span className="truncate text-[10px] font-mono uppercase tracking-[0.22em] text-text-muted">Workspace</span>
            <span className="truncate text-xs font-display font-bold text-text-main">{summary.user.roleLabel}</span>
          </div>
        </div>

        <div className="flex min-w-0 shrink-0 items-center gap-2 sm:gap-3">
          <Link
            prefetch={false}
            href="/profile"
            className="flex min-w-0 items-center gap-3 border border-border-muted bg-background-dark px-2.5 py-2 transition-colors hover:border-primary/60 hover:bg-surface-hover"
          >
            <div className="hidden min-w-0 text-right sm:block">
              <div className="max-w-40 truncate text-sm font-display font-bold text-text-main">{summary.user.fullName}</div>
              <div className="truncate text-[10px] font-mono uppercase tracking-[0.18em] text-text-muted">{summary.user.roleLabel}</div>
            </div>
            <div className="flex h-8 w-8 shrink-0 items-center justify-center border border-border-muted bg-surface-hover text-sm font-display font-bold text-primary">
              {summary.user.initials}
            </div>
          </Link>
          <form action="/api/auth/logout" method="post" className="shrink-0">
            <button
              type="submit"
              className="flex h-10 items-center justify-center border border-border-muted bg-surface px-3 text-[10px] font-mono uppercase tracking-[0.2em] text-text-muted transition-colors hover:border-primary/60 hover:text-primary sm:px-4"
            >
              Logout
            </button>
          </form>
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
          <div className="flex flex-wrap items-start justify-between gap-4 p-6 pb-4">
            <div className="min-w-0">
              <h1 className="text-2xl font-display font-bold text-text-main">{title}</h1>
              <p className="mt-1 max-w-3xl text-sm text-text-muted font-mono leading-6">{description}</p>
            </div>
            {actions ? <div className="flex shrink-0 items-center gap-3">{actions}</div> : null}
          </div>

          <div className="flex-1 overflow-y-auto p-6 pt-2">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
