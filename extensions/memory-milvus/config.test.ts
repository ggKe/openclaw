import fs from "node:fs";
import { describe, expect, it } from "vitest";
import { validateJsonSchemaValue } from "../../src/plugins/schema-validator.js";
import { memoryConfigSchema } from "./config.js";

const manifest = JSON.parse(
  fs.readFileSync(new URL("./openclaw.plugin.json", import.meta.url), "utf-8"),
) as { configSchema: Record<string, unknown> };

describe("memory-milvus config", () => {
  it("accepts milvus config in the manifest schema and preserves it in runtime parsing", () => {
    const manifestResult = validateJsonSchemaValue({
      schema: manifest.configSchema,
      cacheKey: "memory-milvus.manifest.milvus",
      value: {
        embedding: {
          apiKey: "sk-test",
        },
        milvus: {
          address: "localhost:19530",
          collectionName: "test_collection",
          secure: false,
        },
      },
    });

    const parsed = memoryConfigSchema.parse({
      embedding: {
        apiKey: "sk-test",
      },
      milvus: {
        address: "localhost:19530",
        collectionName: "test_collection",
        secure: false,
      },
    });

    expect(manifestResult.ok).toBe(true);
    expect(parsed.milvus.address).toBe("localhost:19530");
    expect(parsed.milvus.collectionName).toBe("test_collection");
    expect(parsed.milvus.secure).toBe(false);
  });

  it("still rejects unrelated unknown top-level config keys", () => {
    expect(() => {
      memoryConfigSchema.parse({
        embedding: {
          apiKey: "sk-test",
        },
        milvus: {
          address: "localhost:19530",
        },
        unexpected: true,
      });
    }).toThrow("memory config has unknown keys: unexpected");
  });

  it("rejects missing milvus.address", () => {
    expect(() => {
      memoryConfigSchema.parse({
        embedding: {
          apiKey: "sk-test",
        },
        milvus: {},
      });
    }).toThrow("milvus.address is required");
  });

  it("accepts milvus authentication options", () => {
    const parsed = memoryConfigSchema.parse({
      embedding: {
        apiKey: "sk-test",
      },
      milvus: {
        address: "localhost:19530",
        username: "testuser",
        password: "testpass",
        token: "testtoken",
      },
    });

    expect(parsed.milvus.username).toBe("testuser");
    expect(parsed.milvus.password).toBe("testpass");
    expect(parsed.milvus.token).toBe("testtoken");
  });

  it("resolves environment variables in milvus config", () => {
    process.env.TEST_MILVUS_ADDRESS = "milvus.example.com:19530";
    process.env.TEST_MILVUS_USER = "envuser";
    process.env.TEST_MILVUS_PASS = "envpass";

    const parsed = memoryConfigSchema.parse({
      embedding: {
        apiKey: "sk-test",
      },
      milvus: {
        address: "${TEST_MILVUS_ADDRESS}",
        username: "${TEST_MILVUS_USER}",
        password: "${TEST_MILVUS_PASS}",
      },
    });

    expect(parsed.milvus.address).toBe("milvus.example.com:19530");
    expect(parsed.milvus.username).toBe("envuser");
    expect(parsed.milvus.password).toBe("envpass");

    delete process.env.TEST_MILVUS_ADDRESS;
    delete process.env.TEST_MILVUS_USER;
    delete process.env.TEST_MILVUS_PASS;
  });

  it("sets default values for optional milvus config", () => {
    const parsed = memoryConfigSchema.parse({
      embedding: {
        apiKey: "sk-test",
      },
      milvus: {
        address: "localhost:19530",
      },
    });

    expect(parsed.milvus.collectionName).toBe("openclaw_memories");
    expect(parsed.milvus.secure).toBe(false);
  });

  it("validates autoCapture and autoRecall defaults", () => {
    const parsed = memoryConfigSchema.parse({
      embedding: {
        apiKey: "sk-test",
      },
      milvus: {
        address: "localhost:19530",
      },
    });

    expect(parsed.autoCapture).toBe(false);
    expect(parsed.autoRecall).toBe(true);
  });

  it("accepts embedding clientType configuration", () => {
    const parsedRest = memoryConfigSchema.parse({
      embedding: {
        apiKey: "sk-test",
        clientType: "rest",
      },
      milvus: {
        address: "localhost:19530",
      },
    });

    const parsedSdk = memoryConfigSchema.parse({
      embedding: {
        apiKey: "sk-test",
        clientType: "openapi-client",
      },
      milvus: {
        address: "localhost:19530",
      },
    });

    expect(parsedRest.embedding.clientType).toBe("rest");
    expect(parsedSdk.embedding.clientType).toBe("openapi-client");
  });

  it("validates captureMaxChars range", () => {
    expect(() => {
      memoryConfigSchema.parse({
        embedding: { apiKey: "sk-test" },
        milvus: { address: "localhost:19530" },
        captureMaxChars: 99,
      });
    }).toThrow("captureMaxChars must be between 100 and 10000");

    expect(() => {
      memoryConfigSchema.parse({
        embedding: { apiKey: "sk-test" },
        milvus: { address: "localhost:19530" },
        captureMaxChars: 10001,
      });
    }).toThrow("captureMaxChars must be between 100 and 10000");

    const parsed = memoryConfigSchema.parse({
      embedding: { apiKey: "sk-test" },
      milvus: { address: "localhost:19530" },
      captureMaxChars: 1500,
    });
    expect(parsed.captureMaxChars).toBe(1500);
  });
});
