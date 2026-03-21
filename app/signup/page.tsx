import { redirect } from 'next/navigation';
import { AuthCredentialForm } from '@/components/auth-credential-form';
import { AuthShell } from '@/components/auth-shell';
import { getAuthenticatedUser } from '@/lib/auth/session';
import { signupPageContent } from '@/content/site';

export const dynamic = 'force-dynamic';

export default async function SignupPage() {
  const user = await getAuthenticatedUser();

  if (user) {
    redirect('/dashboard');
  }

  return (
    <AuthShell
      title={signupPageContent.title}
      subtitle={signupPageContent.subtitle}
    >
      <AuthCredentialForm
        mode="signup"
        fullNameLabel={signupPageContent.form.fullNameLabel}
        fullNamePlaceholder={signupPageContent.form.fullNamePlaceholder}
        emailLabel={signupPageContent.form.emailLabel}
        emailPlaceholder={signupPageContent.form.emailPlaceholder}
        passwordLabel={signupPageContent.form.passwordLabel}
        passwordPlaceholder={signupPageContent.form.passwordPlaceholder}
        submitLabel={signupPageContent.form.submitLabel}
        secondaryPrompt={signupPageContent.secondaryPrompt}
        secondaryHref="/login"
        secondaryLabel={signupPageContent.secondaryLabel}
      />
    </AuthShell>
  );
}
