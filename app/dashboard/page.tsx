import Link from 'next/link';
import { Plus } from 'lucide-react';
import { dashboardContent } from '@/content/site';
import { requireAuthenticatedUser } from '@/lib/auth/session';
import { CreateProjectCard, ProjectCard } from '@/components/project-card';
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
      description={`Found ${projects.length} active assets in current workspace`}
      actions={
        <Link prefetch={false} href="/studio" className="h-9 px-4 flex items-center justify-center gap-2 bg-primary text-background-dark text-sm font-display font-bold hover:bg-primary-hover transition-colors">
          <Plus className="w-4 h-4" />
          {dashboardContent.newGenerationLabel}
        </Link>
      }
    >
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-6 auto-rows-max">
        <CreateProjectCard />
        {projects.map((project) => (
          <ProjectCard key={project.id} project={project} />
        ))}
      </div>
    </WorkspaceShell>
  );
}
