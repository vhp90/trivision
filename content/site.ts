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
  statusLabel: 'Engine v2.4 Online',
  heading: {
    primary: 'Sculpt the Virtual.',
    secondary: 'At the Speed of Thought.',
  },
  description:
    'The next-generation text-to-3D engine. Build immersive worlds, assets, and environments directly in your browser with zero context-switching.',
  primaryCta: {
    href: '/dashboard',
    label: 'Launch Studio',
  },
  terminalLines: [
    '> Initializing WebGL context... [OK]',
    '> Loading neural networks... [OK]',
    '> Awaiting user input_',
  ],
  footer: {
    left: 'SYS.TRIVISION.GEN // ACTIVE',
    right: 'STATUS: OPTIMAL',
  },
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
  panelTitle: 'Text to 3D',
  modelLabel: 'Model',
  promptLabel: 'Prompt',
  clearPromptLabel: 'Clear',
  promptPlaceholder: 'Prompt input is only sent when the selected model supports it.',
  referenceImageLabel: 'Reference Image',
  referenceImageTitle: 'Upload source image',
  referenceImageHelp: 'PNG, JPG, or WEBP up to 10MB',
  replaceImageLabel: 'Replace image',
  lightningPrepTitle: 'Background Removal',
  lightningPrepHelp: 'Run the self-hosted Lightning cleanup step to generate a transparent PNG before TRELLIS.2 generation.',
  lightningPrepAction: 'REMOVE BACKGROUND',
  lightningPrepReadyLabel: 'Background removed',
  lightningPrepPendingLabel: 'Run cleanup to prepare the image for Lightning TRELLIS.2.',
  lightningPrepProcessingLabel: 'Removing background...',
  lightningPrepPreviewTitle: 'Processed Preview',
  generationParametersTitle: 'Generation Parameters',
  generateLabel: 'GENERATE ASSET',
  jobStatusTitle: 'Generation Status',
  sourcePreviewTitle: 'Source Preview',
  maskPreviewTitle: 'Mask Preview',
  segmentationTitle: 'Segmentation',
  segmentationHelp: 'Use point or box prompts to create a MobileSAM mask for this image.',
  generatedPreviewTitle: 'Generated Asset',
  disabledModelBadge: 'Unavailable',
  emptyViewerLabel: 'No completed asset yet',
  emptyViewerHelp: 'Upload an image and start a generation to populate the viewer with a provider-backed 3D result.',
  loaderTitle: 'Generating Asset',
  loaderSubtitle: 'Building mesh, texture, and export package',
  tooltips: {
    model: 'Select the provider-backed model you want to run.',
    prompt: 'Only models that support prompts will send this field to the provider.',
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
