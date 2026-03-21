import type { ReactNode } from 'react';
import Link from 'next/link';
import { Bell, Box, ChevronDown, Search, Settings } from 'lucide-react';
import { brand, dashboardContent } from '@/content/site';
import type { ShellSummary } from '@/lib/db/types';

type WorkspaceShellProps = {
  summary: ShellSummary;
  activeSection: 'dashboard' | 'workspaces' | 'recent-generations' | 'favorites' | 'materials' | 'lighting-rigs' | 'settings';
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
        <div className="flex items-center gap-8">
          <Link href="/" className="flex items-center gap-3 text-text-main">
            <div className="w-5 h-5 text-primary">
              <Box className="w-full h-full" />
            </div>
            <h2 className="text-xl font-display font-bold leading-tight tracking-tight">{brand.name}</h2>
          </Link>

          <label className="hidden md:flex flex-col min-w-40 w-96 h-10">
            <div className="flex w-full flex-1 items-stretch border border-border-muted bg-background-dark focus-within:border-primary transition-colors">
              <div className="text-text-muted flex items-center justify-center pl-3">
                <Search className="w-4 h-4" />
              </div>
              <input
                className="w-full min-w-0 flex-1 bg-transparent text-text-main focus:outline-none focus:ring-0 border-none placeholder:text-text-muted px-3 text-sm font-body"
                placeholder={dashboardContent.searchPlaceholder}
              />
              <div className="flex items-center pr-3 border-l border-border-muted ml-2">
                <button type="button" className="text-text-muted hover:text-text-main px-2 py-1 text-xs font-mono flex items-center gap-1 transition-colors">
                  Sort <ChevronDown className="w-4 h-4" />
                </button>
              </div>
            </div>
          </label>
        </div>

        <div className="flex items-center gap-4">
          <form action="/api/auth/logout" method="post">
            <button type="submit" className="text-[11px] font-mono uppercase tracking-[0.2em] text-text-muted hover:text-primary transition-colors">
              Logout
            </button>
          </form>
          <button type="button" className="text-text-muted hover:text-text-main transition-colors relative">
            <Bell className="w-5 h-5" />
            {summary.user.unreadNotifications > 0 ? (
              <span className="absolute top-0 right-0 w-2 h-2 bg-primary rounded-full border border-surface"></span>
            ) : null}
          </button>
          <div className="h-8 w-8 bg-surface-hover border border-border-muted flex items-center justify-center text-sm font-display font-bold text-primary cursor-pointer hover:border-primary transition-colors">
            {summary.user.initials}
          </div>
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
              <h3 className="text-xs font-mono text-text-muted uppercase tracking-wider mb-2 px-3">{dashboardContent.libraryTitle}</h3>
              {dashboardContent.libraryNav.map((item) => {
                const Icon = item.icon;
                const isActive = activeSection === item.id;
                const count = item.id === 'materials' ? summary.materialCount : summary.lightingRigCount;

                return (
                  <Link
                    key={item.id}
                    href={item.href}
                    className={isActive
                      ? 'flex items-center justify-between px-3 py-2 text-text-main bg-surface-hover transition-colors border-l-2 border-primary'
                      : 'flex items-center justify-between px-3 py-2 text-text-muted hover:bg-surface-hover hover:text-text-main transition-colors border-l-2 border-transparent hover:border-border-muted'}
                  >
                    <div className="flex items-center gap-3">
                      <Icon className={isActive ? 'w-4 h-4 text-primary' : 'w-4 h-4'} />
                      <span className="text-sm font-body font-medium">{item.label}</span>
                    </div>
                    <span className="text-[10px] font-mono bg-border-muted text-text-muted px-1.5 py-0.5">{count}</span>
                  </Link>
                );
              })}
            </div>
          </div>

          <div className="p-4 border-t border-border-muted">
            <Link
              href={dashboardContent.settingsHref}
              className={activeSection === 'settings'
                ? 'flex items-center gap-3 px-3 py-2 bg-surface-hover border-l-2 border-primary text-text-main transition-colors'
                : 'flex items-center gap-3 px-3 py-2 text-text-muted hover:bg-surface-hover hover:text-text-main transition-colors border-l-2 border-transparent hover:border-border-muted'}
            >
              <Settings className={activeSection === 'settings' ? 'w-4 h-4 text-primary' : 'w-4 h-4'} />
              <span className="text-sm font-body font-medium">{dashboardContent.settingsLabel}</span>
            </Link>
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
