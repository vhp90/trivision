import Link from 'next/link';
import { Plus, Sparkles } from 'lucide-react';
import { dashboardContent } from '@/content/site';
import { requireAuthenticatedUser } from '@/lib/auth/session';
import { CreateProjectCard, ProjectCard } from '@/components/project-card';
import { EmptyState } from '@/components/empty-state';
import { WorkspaceShell } from '@/components/workspace-shell';
import { getProjects, getShellSummary } from '@/lib/db/repository';

export const dynamic = 'force-dynamic';

export default async function DashboardPage() {
  const user = await requireAuthenticatedUser();
  const [summary, projects] = await Promise.all([
    getShellSummary(user.id),
    getProjects(user.id, { recentOnly: true }),
  ]);

  return (
    <WorkspaceShell
      summary={summary}
      activeSection="dashboard"
      title={dashboardContent.pageTitle}
      description={`${projects.length} recent ${projects.length === 1 ? 'asset' : 'assets'} in your workspace`}
      actions={
        <Link prefetch={false} href="/studio" className="h-9 px-4 flex items-center justify-center gap-2 bg-primary text-background-dark text-sm font-display font-bold hover:bg-primary-hover transition-colors">
          <Plus className="w-4 h-4" />
          {dashboardContent.newGenerationLabel}
        </Link>
      }
    >
      {projects.length > 0 ? (
        <div className="grid grid-cols-1 auto-rows-max gap-6 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {projects.map((project) => (
            <ProjectCard key={project.id} project={project} />
          ))}
        </div>
      ) : (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
          <EmptyState
            icon={Sparkles}
            title="Start your first generation"
            description="Upload a reference image in the studio and your generated assets will appear here for review, retry, download, and organization."
            actionHref="/studio"
            actionLabel={dashboardContent.newGenerationLabel}
          />
          <CreateProjectCard />
        </div>
      )}
    </WorkspaceShell>
  );
}
