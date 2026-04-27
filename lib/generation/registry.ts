import type {
  GenerationModelDefinition,
  GenerationParameterDefinition,
  GenerationParameterValueMap,
} from '@/lib/generation/types';

function parameter(definition: GenerationParameterDefinition) {
  return definition;
}

export const generationModels: GenerationModelDefinition[] = [
  {
    id: 'microsoft:trellis-2@4b',
    providerId: 'runware:microsoft',
    label: 'TRELLIS.2',
    shortLabel: 'TRELLIS.2',
    description: 'High-fidelity image-to-3D generation from a single reference image.',
    availability: 'enabled',
    capabilities: {
      inputKinds: ['image'],
      promptSupport: 'none',
      outputFormats: ['glb'],
    },
    defaultOutputFormat: 'glb',
    primaryInputLabel: 'Reference Image',
    primaryInputDescription: 'Upload a single source image to generate a textured 3D asset.',
    promptHelperText: 'TRELLIS.2 is image-to-3D only. Prompt-only generation is not supported for this model.',
    parameters: [],
  },
  {
    id: 'lightning:microsoft-trellis-2@4b',
    providerId: 'lightning:microsoft',
    label: 'TRELLIS.2 (Lightning)',
    shortLabel: 'TRELLIS.2 LT',
    description: 'Self-hosted TRELLIS.2 on Lightning AI with built-in background removal before generation.',
    availability: 'enabled',
    capabilities: {
      inputKinds: ['image'],
      promptSupport: 'none',
      outputFormats: ['glb'],
    },
    defaultOutputFormat: 'glb',
    primaryInputLabel: 'Reference Image',
    primaryInputDescription: 'Upload a single source image. The Lightning adapter removes the background before running TRELLIS.2.',
    promptHelperText: 'This self-hosted TRELLIS.2 flow is image-to-3D only. Prompt-only generation is not supported for this model.',
    parameters: [
      parameter({
        key: 'seed',
        label: 'Seed',
        description: 'Reuse a seed for more reproducible results.',
        section: 'Generation',
        type: 'number',
        defaultValue: 42,
        min: 1,
        max: 999999999,
        step: 1,
      }),
      parameter({
        key: 'pipelineType',
        label: 'Pipeline Type',
        description: 'Select the Lightning TRELLIS pipeline preset.',
        section: 'Generation',
        type: 'select',
        defaultValue: '1024_cascade',
        options: [
          { label: '512', value: '512' },
          { label: '1024', value: '1024' },
          { label: '1024 Cascade', value: '1024_cascade' },
          { label: '1536 Cascade', value: '1536_cascade' },
        ],
      }),
      parameter({
        key: 'numSamples',
        label: 'Sample Count',
        description: 'Keep this low unless you know the self-hosted GPU has enough memory.',
        section: 'Generation',
        type: 'number',
        defaultValue: 1,
        min: 1,
        max: 4,
        step: 1,
      }),
      parameter({
        key: 'maxNumTokens',
        label: 'Max Tokens',
        description: 'Upper bound for the Lightning TRELLIS token budget.',
        section: 'Generation',
        type: 'number',
        defaultValue: 49152,
        min: 4096,
        max: 65536,
        step: 1024,
      }),
      parameter({
        key: 'simplifyTarget',
        label: 'Simplify Target',
        description: 'Target triangle budget for the exported mesh.',
        section: 'Mesh Output',
        type: 'number',
        defaultValue: 1000000,
        min: 100000,
        max: 2000000,
        step: 100000,
      }),
      parameter({
        key: 'textureSize',
        label: 'Texture Size',
        description: 'Resolution of the baked texture map.',
        section: 'Mesh Output',
        type: 'select',
        defaultValue: 2048,
        options: [
          { label: '1024', value: 1024 },
          { label: '2048', value: 2048 },
          { label: '3072', value: 3072 },
          { label: '4096', value: 4096 },
        ],
      }),
      parameter({
        key: 'remesh',
        label: 'Remesh',
        description: 'Run mesh cleanup during export.',
        section: 'Mesh Output',
        type: 'boolean',
        defaultValue: true,
      }),
      parameter({
        key: 'remeshBand',
        label: 'Remesh Band',
        description: 'Controls the remeshing band width used during post-processing.',
        section: 'Mesh Output',
        type: 'number',
        defaultValue: 1,
        min: 0,
        max: 4,
        step: 0.1,
      }),
      parameter({
        key: 'remeshProject',
        label: 'Remesh Project',
        description: 'Projection strength applied while remeshing.',
        section: 'Mesh Output',
        type: 'number',
        defaultValue: 0,
        min: 0,
        max: 1,
        step: 0.1,
      }),
    ],
  },
  {
    id: 'meta:sam@3d',
    providerId: 'runware:meta',
    label: 'SAM 3D Objects',
    shortLabel: 'SAM 3D',
    description: 'Single-image 3D reconstruction with mask-guided object extraction.',
    availability: 'enabled',
    capabilities: {
      inputKinds: ['image', 'mask'],
      promptSupport: 'optional',
      outputFormats: ['glb'],
    },
    defaultOutputFormat: 'glb',
    primaryInputLabel: 'Object Image',
    primaryInputDescription: 'Requires both an RGB image and a matching object mask.',
    promptHelperText: 'SAM 3D accepts an optional descriptive prompt. Create a MobileSAM mask from the uploaded image before generating.',
    segmentationSupport: {
      engine: 'mobile-sam',
      required: true,
      promptModes: ['positive-point', 'negative-point', 'box'],
    },
    parameters: [],
  },
];

export function getGenerationModel(modelId: string) {
  return generationModels.find((model) => model.id === modelId) ?? null;
}

export function getDefaultGenerationModel() {
  const enabledModel = generationModels.find((model) => model.availability === 'enabled');

  if (!enabledModel) {
    throw new Error('No enabled generation models are configured.');
  }

  return enabledModel;
}

export function getModelParameterDefaults(model: GenerationModelDefinition): GenerationParameterValueMap {
  return model.parameters.reduce<GenerationParameterValueMap>((accumulator, definition) => {
    accumulator[definition.key] = definition.defaultValue;
    return accumulator;
  }, {});
}

export function groupModelParameters(model: GenerationModelDefinition) {
  const sections = new Map<string, GenerationParameterDefinition[]>();

  for (const parameterDefinition of model.parameters) {
    const section = sections.get(parameterDefinition.section);

    if (section) {
      section.push(parameterDefinition);
      continue;
    }

    sections.set(parameterDefinition.section, [parameterDefinition]);
  }

  return Array.from(sections.entries()).map(([title, parameters]) => ({ title, parameters }));
}
