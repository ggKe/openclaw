import OpenAI from "openai";
import { ensureGlobalUndiciEnvProxyDispatcher } from "openclaw/plugin-sdk/runtime-env";

const DEFAULT_BASE_URL = "https://api.openai.com/v1";

export interface EmbeddingsConfig {
  clientType: "openapi-client" | "rest";
  apiKey: string;
  model: string;
  baseUrl?: string;
  dimensions?: number;
}

/**
 * Embeddings utility class for converting text to vector embeddings.
 * Supports two modes:
 * - "openapi-client": Uses the official OpenAI SDK
 * - "rest": Uses direct HTTP REST requests
 */
export class Embeddings {
  private readonly config: EmbeddingsConfig;
  private openAIClient: OpenAI | null = null;

  constructor(config: EmbeddingsConfig) {
    this.config = {
      ...config,
      baseUrl: config.baseUrl || DEFAULT_BASE_URL,
    };
  }

  /**
   * Embed a single text string into a vector
   */
  async embed(text: string): Promise<number[]> {
    ensureGlobalUndiciEnvProxyDispatcher();

    if (this.config.clientType === "openapi-client") {
      return this.embedWithOpenAIClient(text);
    } else {
      return this.embedWithRest(text);
    }
  }

  /**
   * Embed multiple texts into vectors in batch
   */
  async embedBatch(texts: string[]): Promise<number[][]> {
    ensureGlobalUndiciEnvProxyDispatcher();

    if (this.config.clientType === "openapi-client") {
      return this.embedBatchWithOpenAIClient(texts);
    } else {
      return this.embedBatchWithRest(texts);
    }
  }

  /**
   * Embed using OpenAI SDK client
   */
  private async embedWithOpenAIClient(text: string): Promise<number[]> {
    const client = this.getOpenAIClient();
    const params: OpenAI.Embeddings.EmbeddingCreateParams = {
      model: this.config.model,
      input: text,
    };
    if (this.config.dimensions) {
      params.dimensions = this.config.dimensions;
    }
    const response = await client.embeddings.create(params);
    return response.data[0].embedding;
  }

  /**
   * Embed batch using OpenAI SDK client
   */
  private async embedBatchWithOpenAIClient(texts: string[]): Promise<number[][]> {
    const client = this.getOpenAIClient();
    const params: OpenAI.Embeddings.EmbeddingCreateParams = {
      model: this.config.model,
      input: texts,
    };
    if (this.config.dimensions) {
      params.dimensions = this.config.dimensions;
    }
    const response = await client.embeddings.create(params);
    return response.data.map((item) => item.embedding);
  }

  /**
   * Embed using direct REST request
   */
  private async embedWithRest(text: string): Promise<number[]> {
    const vectors = await this.embedBatchWithRest([text]);
    return vectors[0] || [];
  }

  /**
   * Embed batch using direct REST request
   */
  private async embedBatchWithRest(texts: string[]): Promise<number[][]> {
    const url = `${this.config.baseUrl!.replace(/\/$/, "")}/embeddings`;
    const body: Record<string, unknown> = {
      model: this.config.model,
      input: texts,
    };
    if (this.config.dimensions) {
      body.dimensions = this.config.dimensions;
    }

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.config.apiKey}`,
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Embedding API request failed: ${response.status} ${errorText}`);
    }

    const result = (await response.json()) as {
      data?: Array<{ embedding?: number[] }>;
    };
    const data = result.data ?? [];
    return data.map((entry) => entry.embedding ?? []);
  }

  /**
   * Lazy initialize OpenAI client
   */
  private getOpenAIClient(): OpenAI {
    if (!this.openAIClient) {
      this.openAIClient = new OpenAI({
        apiKey: this.config.apiKey,
        baseURL: this.config.baseUrl,
      });
    }
    return this.openAIClient;
  }
}
