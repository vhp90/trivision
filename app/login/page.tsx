import { redirect } from 'next/navigation';
import { AuthCredentialForm } from '@/components/auth-credential-form';
import { AuthShell } from '@/components/auth-shell';
import { getAuthenticatedUser } from '@/lib/auth/session';
import { loginPageContent } from '@/content/site';

export const dynamic = 'force-dynamic';

export default async function LoginPage() {
  const user = await getAuthenticatedUser();

  if (user) {
    redirect('/dashboard');
  }

  return (
    <AuthShell
      title={loginPageContent.title}
      subtitle={loginPageContent.subtitle}
    >
      <AuthCredentialForm
        mode="login"
        emailLabel={loginPageContent.form.emailLabel}
        emailPlaceholder={loginPageContent.form.emailPlaceholder}
        passwordLabel={loginPageContent.form.passwordLabel}
        passwordPlaceholder={loginPageContent.form.passwordPlaceholder}
        submitLabel={loginPageContent.form.submitLabel}
        secondaryPrompt={loginPageContent.secondaryAction.prompt}
        secondaryHref="/signup"
        secondaryLabel={loginPageContent.secondaryAction.label}
      />
    </AuthShell>
  );
}
