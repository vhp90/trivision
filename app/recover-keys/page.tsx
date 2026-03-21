import { AuthShell } from '@/components/auth-shell';
import { SupportRequestForm } from '@/components/support-request-form';
import { getAuthenticatedUser } from '@/lib/auth/session';
import { recoverKeysPageContent } from '@/content/site';

export const dynamic = 'force-dynamic';

export default async function RecoverKeysPage() {
  const user = await getAuthenticatedUser();

  return (
    <AuthShell
      title={recoverKeysPageContent.title}
      subtitle={recoverKeysPageContent.subtitle}
    >
      <SupportRequestForm
        defaultEmail={user?.email ?? ''}
        emailPlaceholder={recoverKeysPageContent.emailPlaceholder}
        noteLabel={recoverKeysPageContent.noteLabel}
        notePlaceholder={recoverKeysPageContent.notePlaceholder}
        primaryLabel={recoverKeysPageContent.primaryLabel}
        successLabel={recoverKeysPageContent.successLabel}
      />
    </AuthShell>
  );
}
