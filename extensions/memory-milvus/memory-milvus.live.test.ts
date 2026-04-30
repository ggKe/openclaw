/**
 * Memory (Milvus) Plugin Live Tests
 *
 * These tests require:
 * 1. A running Milvus server at localhost:19530
 * 2. OpenAI API key for embeddings
 *
 * Set OPENCLAW_LIVE_TEST=1 and OPENAI_API_KEY to run these tests.
 */

import { afterEach, beforeEach, describe, expect, test } from "vitest";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY ?? "";
const MILVUS_ADDRESS = process.env.MILVUS_ADDRESS ?? "localhost:19530";
const HAS_OPENAI_KEY = Boolean(OPENAI_API_KEY);
const TEST_COLLECTION_NAME = "openclaw_test_memories_" + Date.now();

// Only run live tests if explicitly enabled
const liveEnabled =
  HAS_OPENAI_KEY &&
  process.env.OPENCLAW_LIVE_TEST === "1" &&
  process.env.SKIP_MILVUS_LIVE_TEST !== "1";

const describeLive = liveEnabled ? describe : describe.skip;

describeLive("memory-milvus live tests", () => {
  // Store the collection name so we can clean it up
  let usedCollectionName = TEST_COLLECTION_NAME;

  test("memory tools work end-to-end with real Milvus", async () => {
    const { default: memoryPlugin } = await import("./index.js");
    const liveApiKey = OPENAI_API_KEY;

    // Generate a unique collection name for this test run
    usedCollectionName = `openclaw_test_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;

    // Mock plugin API
    const registeredTools: any[] = [];
    const registeredClis: any[] = [];
    const registeredServices: any[] = [];
    const registeredHooks: Record<string, any[]> = {};
    const logs: string[] = [];

    const mockApi = {
      id: "memory-milvus",
      name: "Memory (Milvus)",
      source: "test",
      config: {},
      pluginConfig: {
        embedding: {
          apiKey: liveApiKey,
          model: "text-embedding-3-small",
          clientType: "openapi-client",
        },
        milvus: {
          address: MILVUS_ADDRESS,
          collectionName: usedCollectionName,
          secure: false,
        },
        autoCapture: false,
        autoRecall: false,
      },
      runtime: {},
      logger: {
        info: (msg: string) => logs.push(`[info] ${msg}`),
        warn: (msg: string) => logs.push(`[warn] ${msg}`),
        error: (msg: string) => logs.push(`[error] ${msg}`),
        debug: (msg: string) => logs.push(`[debug] ${msg}`),
      },
      registerTool: (tool: any, opts: any) => {
        registeredTools.push({ tool, opts });
      },
      registerCli: (registrar: any, opts: any) => {
        registeredClis.push({ registrar, opts });
      },
      registerService: (service: any) => {
        registeredServices.push(service);
      },
      on: (hookName: string, handler: any) => {
        if (!registeredHooks[hookName]) {
          registeredHooks[hookName] = [];
        }
        registeredHooks[hookName].push(handler);
      },
      resolvePath: (p: string) => p,
    };

    console.log(`[test] Using Milvus at ${MILVUS_ADDRESS}, collection: ${usedCollectionName}`);

    // Register plugin - this will trigger lazy initialization
    memoryPlugin.register(mockApi as any);

    // Check registration
    expect(registeredTools.length).toBe(3);
    expect(registeredTools.map((t) => t.opts?.name)).toContain("memory_recall");
    expect(registeredTools.map((t) => t.opts?.name)).toContain("memory_store");
    expect(registeredTools.map((t) => t.opts?.name)).toContain("memory_forget");

    // Get tool functions
    const storeTool = registeredTools.find((t) => t.opts?.name === "memory_store")?.tool;
    const recallTool = registeredTools.find((t) => t.opts?.name === "memory_recall")?.tool;
    const forgetTool = registeredTools.find((t) => t.opts?.name === "memory_forget")?.tool;

    expect(storeTool).toBeDefined();
    expect(recallTool).toBeDefined();
    expect(forgetTool).toBeDefined();

    // ========================================================================
    // Test 1: Store a memory
    // ========================================================================
    console.log("[test] Storing first memory...");
    const storeResult1 = await storeTool.execute("test-call-1", {
      text: "The user prefers dark mode for all applications",
      importance: 0.8,
      category: "preference",
    });

    expect(storeResult1.details?.action).toBe("created");
    const storedId1 = storeResult1.details?.id;
    expect(storedId1).toMatch(/.+/);
    console.log(`[test] Stored memory with ID: ${storedId1}`);

    // ========================================================================
    // Test 2: Store another memory
    // ========================================================================
    console.log("[test] Storing second memory...");
    const storeResult2 = await storeTool.execute("test-call-2", {
      text: "User's email address is developer@example.com",
      importance: 0.9,
      category: "entity",
    });

    expect(storeResult2.details?.action).toBe("created");
    const storedId2 = storeResult2.details?.id;
    expect(storedId2).toBeDefined();
    console.log(`[test] Stored second memory with ID: ${storedId2}`);

    // ========================================================================
    // Test 3: Recall memories - search for dark mode
    // ========================================================================
    console.log("[test] Recalling memories for 'dark mode'...");
    const recallResult1 = await recallTool.execute("test-call-3", {
      query: "dark mode preference",
      limit: 5,
    });

    console.log(`[test] Recall found ${recallResult1.details?.count} memories`);
    expect(recallResult1.details?.count).toBeGreaterThan(0);

    // The first result should be our dark mode memory
    const memories1 = recallResult1.details?.memories ?? [];
    expect(memories1.some((m: any) => m.text.includes("dark mode"))).toBe(true);

    // ========================================================================
    // Test 4: Recall memories - search for email
    // ========================================================================
    console.log("[test] Recalling memories for 'email'...");
    const recallResult2 = await recallTool.execute("test-call-4", {
      query: "email contact",
      limit: 5,
    });

    console.log(`[test] Recall found ${recallResult2.details?.count} memories`);
    expect(recallResult2.details?.count).toBeGreaterThan(0);

    const memories2 = recallResult2.details?.memories ?? [];
    expect(memories2.some((m: any) => m.text.includes("developer@example.com"))).toBe(true);

    // ========================================================================
    // Test 5: Duplicate detection
    // ========================================================================
    console.log("[test] Testing duplicate detection...");
    const duplicateResult = await storeTool.execute("test-call-5", {
      text: "The user prefers dark mode for all applications",
    });

    expect(duplicateResult.details?.action).toBe("duplicate");
    expect(duplicateResult.details?.existingId).toBe(storedId1);
    console.log(`[test] Duplicate correctly detected, matches ID: ${duplicateResult.details?.existingId}`);

    // ========================================================================
    // Test 6: Forget a specific memory
    // ========================================================================
    console.log(`[test] Forgetting memory ${storedId1}...`);
    const forgetResult = await forgetTool.execute("test-call-6", {
      memoryId: storedId1,
    });

    expect(forgetResult.details?.action).toBe("deleted");
    expect(forgetResult.details?.id).toBe(storedId1);
    console.log(`[test] Memory ${storedId1} deleted`);

    // ========================================================================
    // Test 7: Verify the forgotten memory is gone
    // ========================================================================
    console.log("[test] Verifying memory is gone...");
    const recallAfterForget = await recallTool.execute("test-call-7", {
      query: "dark mode preference",
      limit: 5,
    });

    // Should still find some memories (the email one), but not the dark mode one
    const memoriesAfterForget = recallAfterForget.details?.memories ?? [];
    expect(memoriesAfterForget.some((m: any) => m.text.includes("dark mode"))).toBe(false);
    console.log("[test] Confirmed: dark mode memory is no longer found");

    // ========================================================================
    // Test 8: Forget by search (returns candidates)
    // ========================================================================
    console.log("[test] Testing forget by search...");
    const forgetBySearchResult = await forgetTool.execute("test-call-8", {
      query: "email",
    });

    expect(forgetBySearchResult.details?.action).toBe("candidates");
    expect(forgetBySearchResult.details?.candidates).toHaveLength(1);
    console.log(`[test] Found ${forgetBySearchResult.details?.candidates?.length} candidate(s) for deletion`);

    // ========================================================================
    // Cleanup: Delete the remaining memory
    // ========================================================================
    console.log(`[test] Cleaning up remaining memory ${storedId2}...`);
    await forgetTool.execute("test-cleanup", {
      memoryId: storedId2,
    });

    console.log("[test] All tests passed!");
  }, 120000); // 2 minute timeout for live API calls
});

// Cleanup function to drop the test collection if it exists
async function cleanupTestCollection() {
  if (!liveEnabled) {
    return;
  }

  try {
    const { MilvusClient } = await import("@zilliz/milvus2-sdk-node");
    const client = new MilvusClient({
      address: MILVUS_ADDRESS,
    });

    // List collections and drop any test collections
    const collections = await client.listCollections();
    const testCollections = collections.data.filter((c: string) => c.startsWith("openclaw_test_"));

    for (const collection of testCollections) {
      try {
        console.log(`[cleanup] Dropping test collection: ${collection}`);
        await client.dropCollection({ collection_name: collection });
      } catch (e) {
        console.warn(`[cleanup] Failed to drop ${collection}: ${e}`);
      }
    }
  } catch (e) {
    console.warn(`[cleanup] Cleanup failed: ${e}`);
  }
}

// Try to clean up after tests
afterEach(async () => {
  await cleanupTestCollection();
});
