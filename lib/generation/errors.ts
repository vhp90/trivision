const providerSetupPatterns = [
  'RUNWARE_API_KEY is not configured',
  'LIGHTNING_TRELLIS_API_URL is not configured',
  'provider adapter is configured',
];

const providerDownloadPatterns = [
  'without a downloadable asset URL',
  'without a downloadable image URL',
  'could not be downloaded',
  'Unable to download generated asset',
];

const providerTimeoutPatterns = [
  'AbortError',
  'timed out',
  'timeout',
  'fetch failed',
  'timeoutProvider',
];

const providerInputPatterns = [
  'positivePrompt',
  'required parameter',
  'invalid parameter',
  'unsupported parameter',
];

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error ?? '');
}

export function getFriendlyGenerationError(error: unknown) {
  const message = getErrorMessage(error);

  if (providerSetupPatterns.some((pattern) => message.includes(pattern))) {
    return 'Generation provider is not configured. Check the environment settings and try again.';
  }

  if (providerDownloadPatterns.some((pattern) => message.includes(pattern))) {
    return 'The provider finished without a downloadable asset. Retry the generation or adjust the input image.';
  }

  if (providerTimeoutPatterns.some((pattern) => message.toLowerCase().includes(pattern.toLowerCase()))) {
    return 'The provider took too long to respond. Retry the generation in a moment.';
  }

  if (/unauthorized|forbidden|invalid api key/i.test(message)) {
    return 'The provider rejected the request. Check the configured API key and try again.';
  }

  if (/insufficient|payment required|balance|credits/i.test(message)) {
    return 'The provider account does not have enough credits for this generation.';
  }

  if (/rate limit|too many requests|providerRateLimitExceeded|capacity/i.test(message)) {
    return 'The provider is busy right now. Wait a moment, then retry the generation.';
  }

  if (providerInputPatterns.some((pattern) => message.toLowerCase().includes(pattern.toLowerCase()))) {
    return 'The provider could not use those settings. Adjust the prompt or parameters, then retry.';
  }

  return 'Generation failed. Retry with the same input or adjust the image and parameters.';
}
