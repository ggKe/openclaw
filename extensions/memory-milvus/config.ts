export type MemoryConfig = {
  embedding: {
    provider: "openai";
    clientType: "openapi-client" | "rest";
    model: string;
    apiKey: string;
    baseUrl?: string;
    dimensions?: number;
  };
  milvus: {
    address: string;
    username?: string;
    password?: string;
    token?: string;
    collectionName?: string;
    secure?: boolean;
  };
  autoCapture?: boolean;
  autoRecall?: boolean;
  captureMaxChars?: number;
  recallMaxChars?: number;
};

export const MEMORY_CATEGORIES = ["preference", "fact", "decision", "entity", "other"] as const;
export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];

const DEFAULT_MODEL = "text-embedding-3-small";
export const DEFAULT_CAPTURE_MAX_CHARS = 500;
export const DEFAULT_RECALL_MAX_CHARS = 1000;
const DEFAULT_COLLECTION_NAME = "openclaw_memories";

const EMBEDDING_DIMENSIONS: Record<string, number> = {
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
  "Qwen/Qwen3-Embedding-4B": 2560,
};

function assertAllowedKeys(value: Record<string, unknown>, allowed: string[], label: string) {
  const unknown = Object.keys(value).filter((key) => !allowed.includes(key));
  if (unknown.length === 0) {
    return;
  }
  throw new Error(`${label} has unknown keys: ${unknown.join(", ")}`);
}

export function vectorDimsForModel(model: string): number {
  const dims = EMBEDDING_DIMENSIONS[model];
  if (!dims) {
    throw new Error(`Unsupported embedding model: ${model}`);
  }
  return dims;
}

function resolveEnvVars(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, envVar) => {
    const envValue = process.env[envVar];
    if (!envValue) {
      throw new Error(`Environment variable ${envVar} is not set`);
    }
    return envValue;
  });
}

function resolveEmbeddingModel(embedding: Record<string, unknown>): string {
  const model = typeof embedding.model === "string" ? embedding.model : DEFAULT_MODEL;
  if (typeof embedding.dimensions !== "number") {
    vectorDimsForModel(model);
  }
  return model;
}

function resolveClientType(embedding: Record<string, unknown>): "openapi-client" | "rest" {
  const clientType = embedding.clientType;
  if (clientType === "openapi-client" || clientType === "rest") {
    return clientType;
  }
  return "openapi-client";
}

export const memoryConfigSchema = {
  parse(value: unknown): MemoryConfig {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error("memory config required");
    }
    const cfg = value as Record<string, unknown>;
    assertAllowedKeys(
      cfg,
      ["embedding", "milvus", "autoCapture", "autoRecall", "captureMaxChars", "recallMaxChars"],
      "memory config",
    );

    // Embedding config
    const embedding = cfg.embedding as Record<string, unknown> | undefined;
    if (!embedding || typeof embedding.apiKey !== "string") {
      throw new Error("embedding.apiKey is required");
    }
    assertAllowedKeys(embedding, ["apiKey", "model", "baseUrl", "dimensions", "clientType"], "embedding config");

    const model = resolveEmbeddingModel(embedding);
    const clientType = resolveClientType(embedding);

    // Milvus config
    const milvus = cfg.milvus as Record<string, unknown> | undefined;
    if (!milvus || typeof milvus.address !== "string") {
      throw new Error("milvus.address is required");
    }
    assertAllowedKeys(
      milvus,
      ["address", "username", "password", "token", "collectionName", "secure"],
      "milvus config",
    );

    // Validate authentication configuration
    const username = typeof milvus.username === "string" ? milvus.username.trim() : "";
    const password = typeof milvus.password === "string" ? milvus.password.trim() : "";
    const token = typeof milvus.token === "string" ? milvus.token.trim() : "";

    // Check that we don't have conflicting auth methods
    if (token && (username || password)) {
      throw new Error("milvus: cannot provide both token and username/password authentication");
    }

    // Check that username and password are both provided if either is provided
    if (username && !password) {
      throw new Error("milvus: password is required when username is provided");
    }
    if (password && !username) {
      throw new Error("milvus: username is required when password is provided");
    }

    const captureMaxChars =
      typeof cfg.captureMaxChars === "number" ? Math.floor(cfg.captureMaxChars) : undefined;
    if (
      typeof captureMaxChars === "number" &&
      (captureMaxChars < 100 || captureMaxChars > 10_000)
    ) {
      throw new Error("captureMaxChars must be between 100 and 10000");
    }

    const recallMaxChars =
      typeof cfg.recallMaxChars === "number" ? Math.floor(cfg.recallMaxChars) : undefined;
    if (typeof recallMaxChars === "number" && (recallMaxChars < 100 || recallMaxChars > 10_000)) {
      throw new Error("recallMaxChars must be between 100 and 10000");
    }

    return {
      embedding: {
        provider: "openai",
        clientType,
        model,
        apiKey: resolveEnvVars(embedding.apiKey),
        baseUrl:
          typeof embedding.baseUrl === "string" ? resolveEnvVars(embedding.baseUrl) : undefined,
        dimensions: typeof embedding.dimensions === "number" ? embedding.dimensions : undefined,
      },
      milvus: {
        address: resolveEnvVars(milvus.address),
        username: typeof milvus.username === "string" && milvus.username.trim() !== "" ? resolveEnvVars(milvus.username) : undefined,
        password: typeof milvus.password === "string" && milvus.password.trim() !== "" ? resolveEnvVars(milvus.password) : undefined,
        token: typeof milvus.token === "string" && milvus.token.trim() !== "" ? resolveEnvVars(milvus.token) : undefined,
        collectionName:
          typeof milvus.collectionName === "string"
            ? milvus.collectionName
            : DEFAULT_COLLECTION_NAME,
        secure: milvus.secure === true,
      },
      autoCapture: cfg.autoCapture === true,
      autoRecall: cfg.autoRecall !== false,
      captureMaxChars: captureMaxChars ?? DEFAULT_CAPTURE_MAX_CHARS,
      recallMaxChars: recallMaxChars ?? DEFAULT_RECALL_MAX_CHARS,
    };
  },
  uiHints: {
    "embedding.apiKey": {
      label: "OpenAI API Key",
      sensitive: true,
      placeholder: "sk-proj-...",
      help: "API key for OpenAI embeddings (or use ${OPENAI_API_KEY})",
    },
    "embedding.baseUrl": {
      label: "Base URL",
      placeholder: "https://api.openai.com/v1",
      help: "Base URL for compatible providers (e.g. http://localhost:11434/v1)",
      advanced: true,
    },
    "embedding.dimensions": {
      label: "Dimensions",
      placeholder: "1536",
      help: "Vector dimensions for custom models (required for non-standard models)",
      advanced: true,
    },
    "embedding.model": {
      label: "Embedding Model",
      placeholder: DEFAULT_MODEL,
      help: "OpenAI embedding model to use",
    },
    "embedding.clientType": {
      label: "Client Type",
      placeholder: "openapi-client",
      help: "Client type to use for embeddings: 'openapi-client' (uses OpenAI SDK) or 'rest' (uses direct HTTP requests)",
      advanced: true,
    },
    "milvus.address": {
      label: "Milvus Address",
      placeholder: "localhost:19530",
      help: "Milvus server address (host:port)",
    },
    "milvus.collectionName": {
      label: "Collection Name",
      placeholder: DEFAULT_COLLECTION_NAME,
      help: "Milvus collection name for storing memories",
      advanced: true,
    },
    "milvus.secure": {
      label: "Use TLS",
      help: "Enable TLS/SSL for Milvus connection",
      advanced: true,
    },
    autoCapture: {
      label: "Auto-Capture",
      help: "Automatically capture important information from conversations",
    },
    autoRecall: {
      label: "Auto-Recall",
      help: "Automatically inject relevant memories into context",
    },
    captureMaxChars: {
      label: "Capture Max Chars",
      help: "Maximum message length eligible for auto-capture",
      advanced: true,
      placeholder: String(DEFAULT_CAPTURE_MAX_CHARS),
    },
    recallMaxChars: {
      label: "Recall Query Max Chars",
      help: "Maximum prompt/query length embedded for memory recall. Lower for small local embedding models.",
      advanced: true,
      placeholder: String(DEFAULT_RECALL_MAX_CHARS),
    },
  },
};
