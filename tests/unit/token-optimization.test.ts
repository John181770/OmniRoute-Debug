import assert from "node:assert";
import { test } from "node:test";
import { claudeToOpenAIRequest } from "../../open-sse/translator/request/claude-to-openai.ts";
import { coerceSchemaNumericFields } from "../../open-sse/translator/helpers/schemaCoercion.ts";

test("claudeToOpenAIRequest: deduplicates identical system prompt blocks", () => {
  const body = {
    system: [
      "You are a helpful coding assistant.",
      "You are a helpful coding assistant.",
      "Always output JSON.",
    ],
    messages: [{ role: "user", content: "Hello" }],
  };

  const result = claudeToOpenAIRequest("gpt-4o", body, false);
  const systemMsg = result.messages.find((m) => m.role === "system");

  assert.ok(systemMsg);
  assert.strictEqual(systemMsg.content, "You are a helpful coding assistant.\nAlways output JSON.");
});

test("claudeToOpenAIRequest: retrenches base64 images in historical turns while keeping latest turn image", () => {
  const body = {
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "Here is an OCR screenshot of my document." },
          {
            type: "image",
            source: { type: "base64", media_type: "image/png", data: "A".repeat(150000) },
          },
        ],
      },
      {
        role: "assistant",
        content: "I see the document text.",
      },
      {
        role: "user",
        content: "Can you summarize paragraph 2?",
      },
    ],
  };

  const result = claudeToOpenAIRequest("gpt-4o", body, false);

  // Turn 0 (historical user turn) image should be replaced with placeholder
  const firstUserMsg = result.messages.find((m) => m.role === "user");
  assert.ok(firstUserMsg);
  const contentStr =
    typeof firstUserMsg.content === "string"
      ? firstUserMsg.content
      : JSON.stringify(firstUserMsg.content);
  assert.ok(contentStr.includes("[Prior attached image retained in context]"));

  // Check total payload size reduction
  const payloadStr = JSON.stringify(result);
  assert.ok(
    payloadStr.length < 5000,
    `Payload size ${payloadStr.length} is larger than expected 5000 bytes`
  );
});

test("claudeToOpenAIRequest: retains base64 image on single turn request", () => {
  const body = {
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "Analyze this image." },
          {
            type: "image",
            source: { type: "base64", media_type: "image/png", data: "B".repeat(1000) },
          },
        ],
      },
    ],
  };

  const result = claudeToOpenAIRequest("gpt-4o", body, false);
  const userMsg = result.messages[0];

  assert.ok(Array.isArray(userMsg.content));
  const imagePart = (userMsg.content as Array<{ type: string; image_url?: { url: string } }>).find(
    (p) => p.type === "image_url"
  );
  assert.ok(imagePart);
  assert.ok(imagePart.image_url?.url.includes("B".repeat(1000)));
});

test("coerceSchemaNumericFields: strips non-functional JSON schema metadata", () => {
  const schema = {
    $schema: "http://json-schema.org/draft-07/schema#",
    title: "FileInput",
    $comment: "Internal tool schema",
    type: "object",
    properties: {
      filePath: {
        type: "string",
        description: "Path to file",
        examples: ["/src/main.ts"],
        default: "/src/index.ts",
      },
    },
  };

  const cleaned = coerceSchemaNumericFields(schema) as {
    $schema?: string;
    title?: string;
    $comment?: string;
    properties?: {
      filePath?: {
        type?: string;
        description?: string;
        examples?: string[];
        default?: string;
      };
    };
  };

  assert.strictEqual(cleaned.$schema, undefined);
  assert.strictEqual(cleaned.title, undefined);
  assert.strictEqual(cleaned.$comment, undefined);
  assert.strictEqual(cleaned.properties?.filePath?.examples, undefined);
  assert.strictEqual(cleaned.properties?.filePath?.default, undefined);
  assert.strictEqual(cleaned.properties?.filePath?.type, "string");
});
