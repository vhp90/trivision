import {
  Folder,
  History,
  LayoutDashboard,
  Settings,
  Star,
  UserRound,
  type LucideIcon,
} from 'lucide-react';
import { runtimeDefaults, studioDefaults } from '@/lib/config/app';

export const brand = {
  name: 'Trivision',
  uppercaseName: 'TRIVISION',
  title: 'Trivision - Generate 3D Assets',
  description: 'Professional-grade text-to-3D workspace directly in your browser.',
} as const;

export const landingPageContent = {
  statusLabel: '3D Asset Generation Workspace',
  heading: {
    primary: 'Trivision',
    secondary: 'Turn reference images into review-ready 3D assets.',
  },
  description:
    'Upload a source image, prepare the subject, generate a 3D asset, and manage every result from one focused studio interface.',
  primaryCta: {
    href: '/signup',
    label: 'Start Creating',
  },
  secondaryCta: {
    href: '/login',
    label: 'Sign In',
  },
  workflow: [
    {
      title: 'Import',
      description: 'Start from a product shot, concept frame, or object reference.',
    },
    {
      title: 'Prepare',
      description: 'Clean the subject with background removal or mask-guided selection.',
    },
    {
      title: 'Generate',
      description: 'Run provider-backed 3D generation with model-specific controls.',
    },
    {
      title: 'Review',
      description: 'Preview, download, favorite, retry, or rename the generated asset.',
    },
  ],
  capabilities: [
    {
      eyebrow: 'Studio',
      title: 'A focused workspace for asset iteration',
      description:
        'Trivision keeps upload, masking, generation controls, status tracking, and preview in one visual flow so each attempt is easy to understand and repeat.',
    },
    {
      eyebrow: 'Pipeline',
      title: 'Built for image-to-3D workflows',
      description:
        'Reference images, optional segmentation masks, Lightning preprocessing, and model parameters stay attached to the generated asset record.',
    },
    {
      eyebrow: 'Library',
      title: 'Generated assets stay manageable',
      description:
        'Recent work, favorites, retries, downloads, and project updates are available from the dashboard and studio views.',
    },
  ],
  modelSurfaces: [
    'TRELLIS.2',
    'Lightning TRELLIS.2',
    'SAM 3D Objects',
  ],
} as const;

export const loginPageContent = {
  title: 'Authenticate',
  subtitle: 'Secure access to Trivision Studio Workspace',
  form: {
    emailLabel: 'Email Address',
    emailPlaceholder: 'artist@studio.com',
    passwordLabel: 'Password',
    passwordPlaceholder: '••••••••••••',
    submitLabel: 'Authenticate',
  },
  secondaryAction: {
    label: 'Sign Up',
    prompt: 'New to Trivision?',
  },
} as const;

export type DashboardNavItem = {
  id: string;
  label: string;
  count?: string;
  href: string;
  icon: LucideIcon;
};

export const dashboardContent: {
  workspaceTitle: string;
  workspaceNav: DashboardNavItem[];
  accountTitle: string;
  accountNav: DashboardNavItem[];
  pageTitle: string;
  newGenerationLabel: string;
  createCardLabel: string;
} = {
  workspaceTitle: 'Workspace',
  workspaceNav: [
    {
      id: 'dashboard',
      label: 'Dashboard',
      href: '/dashboard',
      icon: LayoutDashboard,
    },
    {
      id: 'workspaces',
      label: 'Workspaces',
      href: '/workspaces',
      icon: Folder,
    },
    {
      id: 'recent-generations',
      label: 'Recent Generations',
      href: '/recent-generations',
      icon: History,
    },
    {
      id: 'favorites',
      label: 'Favorites',
      href: '/favorites',
      icon: Star,
    },
  ],
  accountTitle: 'Account',
  accountNav: [
    {
      id: 'profile',
      label: 'Profile',
      href: '/profile',
      icon: UserRound,
    },
    {
      id: 'settings',
      label: 'Settings',
      href: '/settings',
      icon: Settings,
    },
  ],
  pageTitle: 'Recent Projects',
  newGenerationLabel: 'New Generation',
  createCardLabel: 'Start Generation',
};

export const signupPageContent = {
  title: 'Create Account',
  subtitle: 'Create your Trivision Studio workspace account.',
  form: {
    fullNameLabel: 'Name',
    fullNamePlaceholder: 'Your full name',
    emailLabel: 'Email Address',
    emailPlaceholder: 'artist@studio.com',
    passwordLabel: 'Password',
    passwordPlaceholder: 'Create a strong password',
    submitLabel: 'Create Account',
  },
  secondaryPrompt: 'Already have access?',
  secondaryLabel: 'Log In',
} as const;

export const collectionPageContent = {
  workspaces: {
    title: 'Workspaces',
    description: 'Organize active pipelines, experiments, and review-ready assets in one workspace.',
  },
  recent: {
    title: 'Recent Generations',
    description: 'Browse your latest generated assets and continue from previous results.',
  },
  favorites: {
    title: 'Favorites',
    description: 'Pinned assets for quick access during review and iteration.',
  },
  profile: {
    title: 'Profile',
    description: 'Manage the identity and account details attached to your studio workspace.',
  },
  settings: {
    title: 'Settings',
    description: 'Review and update workspace defaults for generation and export.',
  },
} as const;

export const studioContent = {
  statusLabel: 'Engine Ready',
  exportLabel: 'Export',
  panelTitle: 'Image to 3D',
  modelLabel: 'Model',
  promptLabel: 'Prompt',
  clearPromptLabel: 'Clear',
  promptPlaceholder: 'Describe the asset or selected object.',
  referenceImageLabel: 'Reference Image',
  referenceImageTitle: 'Upload source image',
  referenceImageHelp: 'PNG, JPG, or WEBP up to 10MB',
  replaceImageLabel: 'Replace image',
  generationParametersTitle: 'Generation Parameters',
  generateLabel: 'GENERATE ASSET',
  jobStatusTitle: 'Generation Status',
  sourcePreviewTitle: 'Source Preview',
  maskPreviewTitle: 'Mask Preview',
  segmentationTitle: 'Segmentation',
  segmentationHelp: 'Use point or box prompts to create a MobileSAM mask for this image.',
  generatedPreviewTitle: 'Generated Asset',
  disabledModelBadge: 'Unavailable',
  emptyViewerLabel: 'Ready for generation',
  emptyViewerHelp: 'Upload a reference image, tune the model parameters, and start a 3D generation.',
  loaderTitle: 'Generating Asset',
  loaderSubtitle: 'Building mesh, texture, and export package',
  tooltips: {
    model: 'Select the provider-backed model you want to run.',
    prompt: 'Describe the source or target asset.',
    sourceImage: 'Upload the reference image used for the current generation.',
    wireframe: 'Switch to the wireframe viewport overlay.',
    solid: 'Switch back to the solid viewport preview.',
    cameraReset: 'Reset the viewer to the default camera framing.',
    lighting: 'Toggle the viewport lighting intensity.',
    download: 'Download the generated asset in its stored output format.',
  },
  viewerMetrics: [
    { label: 'Tris', value: studioDefaults.emptyMetrics.tris },
    { label: 'Verts', value: studioDefaults.emptyMetrics.verts },
    { label: 'FPS', value: studioDefaults.emptyMetrics.fps, colorClassName: 'text-success' },
  ],
  overlay: {
    compilerTitle: 'Compiler Log',
    lines: [
      `> Initializing Trivision Engine ${runtimeDefaults.engineVersion}`,
      '> Validating prompt payload...',
      '> Preparing workspace record...',
      '> Allocating draft geometry buffers...',
    ],
    progressPrefix: '> Generating base voxel mesh',
    progressValue: '[||||||    ] 60%',
  },
  propertiesTitle: 'Properties',
  geometryTitle: 'Geometry',
  scaleLabels: ['Scale X', 'Scale Y', 'Scale Z'],
  scaleValue: '1.000',
  autoUvLabel: 'Auto-UV Unwrap',
  materialsTitle: 'Materials',
  environmentTitle: 'Environment',
  exportConfigTitle: 'Export Config',
} as const;
