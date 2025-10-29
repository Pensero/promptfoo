import { OpenAiChatCompletionProvider } from './openai/chat';
import { OpenAiCompletionProvider } from './openai/completion';
import { OpenAiEmbeddingProvider } from './openai/embedding';

import type { ApiProvider, ProviderOptions } from '../types/index';
import type { EnvOverrides } from '../types/env';
import { maybeLoadFromExternalFile } from '../util/file';


/**
 * Creates a TogetherAI provider using OpenAI-compatible endpoints
 *
 * TogetherAI supports many parameters beyond standard OpenAI ones.
 * All parameters are automatically passed through to the TogetherAI API.
 */
export function createTogetherAiProvider(
  providerPath: string,
  options: {
    config?: ProviderOptions;
    id?: string;
    env?: EnvOverrides;
  } = {},
): ApiProvider {
  const splits = providerPath.split(':');

  const config = options.config?.config || {};
  
  // Extract configs that need special handling or are client-side only
  const { showThinking, response_format, ...passthroughConfig } = config;

  let togetherApiResponseFormat: unknown;
  
  if (
    response_format !== undefined && response_format !== null) {
    // Load from file if it's a file:// reference string
    const loadedFormat = maybeLoadFromExternalFile(response_format);
    
    // Convert OpenAI json_schema format to TogetherAI format
    if (
      typeof loadedFormat === 'object' &&
      loadedFormat !== null &&
      'type' in loadedFormat &&
      loadedFormat.type === 'json_schema'
    ) {
      const formatObj = loadedFormat as Record<string, unknown>;
      
      // Extract the actual schema from various possible structures:
      // - { schema: {...} } -> use schema
      // - { json_schema: { schema: {...} } } -> use json_schema.schema (OpenAI format)
      // - { json_schema: {...} } -> use json_schema (already TogetherAI format)
      let actualSchema: unknown;
      
      if ('json_schema' in formatObj && typeof formatObj.json_schema === 'object' && formatObj.json_schema !== null) {
        const jsonSchemaObj = formatObj.json_schema as Record<string, unknown>;
        actualSchema = 'schema' in jsonSchemaObj ? jsonSchemaObj.schema : formatObj.json_schema;
      } else if ('schema' in formatObj) {
        actualSchema = formatObj.schema;
      }
      
      if (actualSchema !== undefined) {
        togetherApiResponseFormat = {
          type: 'json_schema',
          json_schema: actualSchema,
        };
      }
    }
  }
  
  const togetherAiConfig = {
    ...options,
    config: {
      apiBaseUrl: 'https://api.together.xyz/v1',
      apiKeyEnvar: 'TOGETHER_API_KEY',
      // Client-side config
      ...(showThinking !== undefined && { showThinking }),
      // Keep the original response_format for the client
      ...(response_format !== undefined && { response_format }),
      // Everything else goes to passthrough for the API
      passthrough: {
        ...(togetherApiResponseFormat !== undefined && { response_format: togetherApiResponseFormat }),
        ...passthroughConfig,
      },
    },
  };

  if (splits[1] === 'chat') {
    const modelName = splits.slice(2).join(':');
    return new OpenAiChatCompletionProvider(modelName, togetherAiConfig);
  } else if (splits[1] === 'completion') {
    const modelName = splits.slice(2).join(':');
    return new OpenAiCompletionProvider(modelName, togetherAiConfig);
  } else if (splits[1] === 'embedding' || splits[1] === 'embeddings') {
    const modelName = splits.slice(2).join(':');
    return new OpenAiEmbeddingProvider(modelName, togetherAiConfig);
  } else {
    // If no specific type is provided, default to chat
    const modelName = splits.slice(1).join(':');
    return new OpenAiChatCompletionProvider(modelName, togetherAiConfig);
  }
}
