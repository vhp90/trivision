import Link from 'next/link';
import type { LucideIcon } from 'lucide-react';

type EmptyStateProps = {
  icon: LucideIcon;
  title: string;
  description: string;
  actionHref?: string;
  actionLabel?: string;
};

export function EmptyState({
  icon: Icon,
  title,
  description,
  actionHref,
  actionLabel,
}: EmptyStateProps) {
  return (
    <div className="border border-dashed border-border-muted bg-surface/50 p-8">
      <div className="flex max-w-xl flex-col gap-5">
        <div className="flex h-11 w-11 items-center justify-center border border-border-muted bg-background-dark text-primary">
          <Icon className="h-5 w-5" />
        </div>
        <div>
          <h2 className="text-xl font-display font-bold text-text-main">{title}</h2>
          <p className="mt-2 text-sm leading-6 text-text-muted">{description}</p>
        </div>
        {actionHref && actionLabel ? (
          <Link
            href={actionHref}
            prefetch={false}
            className="inline-flex h-10 w-fit items-center justify-center bg-primary px-4 text-sm font-display font-bold text-background-dark transition-colors hover:bg-primary-hover"
          >
            {actionLabel}
          </Link>
        ) : null}
      </div>
    </div>
  );
}
