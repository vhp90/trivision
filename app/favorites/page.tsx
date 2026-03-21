import Link from 'next/link';
import { Star } from 'lucide-react';
import { collectionPageContent } from '@/content/site';
import { requireAuthenticatedUser } from '@/lib/auth/session';
import { ProjectCard } from '@/components/project-card';
import { WorkspaceShell } from '@/components/workspace-shell';
import { getProjects, getShellSummary } from '@/lib/db/repository';

export const dynamic = 'force-dynamic';

export default async function FavoritesPage() {
  const user = await requireAuthenticatedUser();
  const [summary, projects] = await Promise.all([
    getShellSummary(user.id),
    getProjects(user.id, { favoritesOnly: true }),
  ]);

  return (
    <WorkspaceShell
      summary={summary}
      activeSection="favorites"
      title={collectionPageContent.favorites.title}
      description={collectionPageContent.favorites.description}
      actions={
        <Link href="/dashboard" className="h-9 px-4 flex items-center justify-center gap-2 border border-border-muted bg-surface text-text-main text-sm font-body hover:bg-surface-hover hover:border-primary/50 transition-colors">
          <Star className="w-4 h-4" />
          Back to Dashboard
        </Link>
      }
    >
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-6">
        {projects.map((project) => (
          <ProjectCard key={project.id} project={project} />
        ))}
      </div>
    </WorkspaceShell>
  );
}
