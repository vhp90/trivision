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
        key: 'settings.textureSize',
        label: 'Texture Size',
        description: 'Resolution of the generated texture map.',
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
        key: 'settings.resolution',
        label: 'Voxel Resolution',
        description: 'Underlying voxel grid resolution for geometry generation.',
        section: 'Mesh Output',
        type: 'select',
        defaultValue: 1024,
        options: [
          { label: '512', value: 512 },
          { label: '1024', value: 1024 },
          { label: '1536', value: 1536 },
        ],
      }),
      parameter({
        key: 'settings.decimationTarget',
        label: 'Decimation Target',
        description: 'Target polygon count for a lighter output mesh.',
        section: 'Mesh Output',
        type: 'number',
        defaultValue: 500000,
        min: 100000,
        max: 1000000,
        step: 100000,
      }),
      parameter({
        key: 'settings.remesh',
        label: 'Remesh',
        description: 'Clean up the topology before exporting the final asset.',
        section: 'Mesh Output',
        type: 'boolean',
        defaultValue: true,
      }),
      parameter({
        key: 'settings.sparseStructure.guidanceStrength',
        label: 'Guidance Strength',
        description: 'Controls how strongly the sparse structure stage follows the input.',
        section: 'Sparse Structure',
        type: 'number',
        defaultValue: 7.5,
        min: 1,
        max: 10,
        step: 0.1,
      }),
      parameter({
        key: 'settings.sparseStructure.guidanceRescale',
        label: 'Guidance Rescale',
        description: 'Balances guidance strength at the sparse structure stage.',
        section: 'Sparse Structure',
        type: 'number',
        defaultValue: 0.5,
        min: 0,
        max: 1,
        step: 0.1,
      }),
      parameter({
        key: 'settings.sparseStructure.steps',
        label: 'Steps',
        description: 'Inference steps for the sparse structure stage.',
        section: 'Sparse Structure',
        type: 'number',
        defaultValue: 25,
        min: 1,
        max: 50,
        step: 1,
      }),
      parameter({
        key: 'settings.sparseStructure.rescaleT',
        label: 'Rescale T',
        description: 'Stage-specific rescale parameter for sparse structure.',
        section: 'Sparse Structure',
        type: 'number',
        defaultValue: 2,
        min: 1,
        max: 6,
        step: 1,
      }),
      parameter({
        key: 'settings.shapeSlat.guidanceStrength',
        label: 'Guidance Strength',
        description: 'Controls geometry refinement during the Shape SLAT stage.',
        section: 'Shape SLAT',
        type: 'number',
        defaultValue: 5,
        min: 1,
        max: 10,
        step: 0.1,
      }),
      parameter({
        key: 'settings.shapeSlat.guidanceRescale',
        label: 'Guidance Rescale',
        description: 'Balances guidance strength during Shape SLAT refinement.',
        section: 'Shape SLAT',
        type: 'number',
        defaultValue: 0.5,
        min: 0,
        max: 1,
        step: 0.1,
      }),
      parameter({
        key: 'settings.shapeSlat.steps',
        label: 'Steps',
        description: 'Inference steps for the Shape SLAT stage.',
        section: 'Shape SLAT',
        type: 'number',
        defaultValue: 30,
        min: 1,
        max: 50,
        step: 1,
      }),
      parameter({
        key: 'settings.shapeSlat.rescaleT',
        label: 'Rescale T',
        description: 'Stage-specific rescale parameter for Shape SLAT.',
        section: 'Shape SLAT',
        type: 'number',
        defaultValue: 2,
        min: 1,
        max: 6,
        step: 1,
      }),
      parameter({
        key: 'settings.texSlat.guidanceStrength',
        label: 'Guidance Strength',
        description: 'Controls texture generation during the Texture SLAT stage.',
        section: 'Texture SLAT',
        type: 'number',
        defaultValue: 6,
        min: 1,
        max: 10,
        step: 0.1,
      }),
      parameter({
        key: 'settings.texSlat.guidanceRescale',
        label: 'Guidance Rescale',
        description: 'Balances guidance strength during the texture stage.',
        section: 'Texture SLAT',
        type: 'number',
        defaultValue: 0.5,
        min: 0,
        max: 1,
        step: 0.1,
      }),
      parameter({
        key: 'settings.texSlat.steps',
        label: 'Steps',
        description: 'Inference steps for the Texture SLAT stage.',
        section: 'Texture SLAT',
        type: 'number',
        defaultValue: 20,
        min: 1,
        max: 50,
        step: 1,
      }),
      parameter({
        key: 'settings.texSlat.rescaleT',
        label: 'Rescale T',
        description: 'Stage-specific rescale parameter for Texture SLAT.',
        section: 'Texture SLAT',
        type: 'number',
        defaultValue: 2,
        min: 1,
        max: 6,
        step: 1,
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
    parameters: [
      parameter({
        key: 'seed',
        label: 'Seed',
        description: 'Reuse a seed for more reproducible reconstruction results.',
        section: 'Generation',
        type: 'number',
        defaultValue: 42,
        min: 1,
        max: 999999999,
        step: 1,
      }),
    ],
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
