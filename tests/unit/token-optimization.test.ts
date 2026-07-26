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

test("claudeToOpenAIRequest: simple chat ('hi') yields 0 tools, 1 system message, and < 1.2K prompt tokens", () => {
  const sampleTools = Array.from({ length: 20 }, (_, i) => ({
    name: `tool_${i}`,
    description: `Detailed description for tool ${i} with extra documentation text.`,
    input_schema: {
      type: "object",
      properties: { arg: { type: "string", description: "arg desc" } },
    },
  }));

  const body = {
    system: "You are a helpful assistant.",
    messages: [{ role: "user", content: "hi" }],
    tools: sampleTools,
  };

  const result = claudeToOpenAIRequest("gpt-4o", body, false);
  const payload = JSON.stringify(result);
  const approxTokens = Math.ceil(payload.length / 4);

  assert.strictEqual(result.tools, undefined, "Tools should be omitted for simple chat");
  assert.strictEqual(
    result.tool_choice,
    undefined,
    "Tool choice should be omitted for simple chat"
  );
  const systemCount = result.messages.filter((m) => m.role === "system").length;
  assert.strictEqual(systemCount, 1, "Should have exactly 1 system message");
  assert.ok(approxTokens < 1200, `Approx tokens ${approxTokens} is not under target 1200`);
});

test("claudeToOpenAIRequest: tool request ('Search Google') injects only required tools (< 6K tokens)", () => {
  const sampleTools = Array.from({ length: 20 }, (_, i) => ({
    name: i === 5 ? "web_search" : `tool_${i}`,
    description: i === 5 ? "Search Google for live web results." : `Tool description ${i}`,
    input_schema: { type: "object", properties: { query: { type: "string" } } },
  }));

  const body = {
    system: "You are a helpful assistant.",
    messages: [{ role: "user", content: "Search Google for latest news" }],
    tools: sampleTools,
  };

  const result = claudeToOpenAIRequest("gpt-4o", body, false);
  const payload = JSON.stringify(result);
  const approxTokens = Math.ceil(payload.length / 4);

  assert.ok(Array.isArray(result.tools));
  assert.strictEqual(result.tools.length, 1, "Should inject only the web_search tool");
  assert.strictEqual((result.tools[0].function as { name: string }).name, "web_search");
  assert.ok(approxTokens < 6000, `Approx tokens ${approxTokens} is not under target 6000`);
});

test("claudeToOpenAIRequest: merges system prompt from body.system and body.messages into single system message", () => {
  const body = {
    system: ["Instruction set A.", "Instruction set B."],
    messages: [
      { role: "system", content: "Instruction set C." },
      { role: "user", content: "Hello" },
    ],
  };

  const result = claudeToOpenAIRequest("gpt-4o", body, false);
  const systemMessages = result.messages.filter((m) => m.role === "system");

  assert.strictEqual(systemMessages.length, 1);
  assert.strictEqual(
    systemMessages[0].content,
    "Instruction set A.\nInstruction set B.\nInstruction set C."
  );
});

test("claudeToOpenAIRequest: removes empty messages and deduplicates adjacent identical turns", () => {
  const body = {
    messages: [
      { role: "user", content: "hello" },
      { role: "user", content: "hello" },
      { role: "assistant", content: "" },
      { role: "user", content: "how are you?" },
    ],
  };

  const result = claudeToOpenAIRequest("gpt-4o", body, false);
  assert.strictEqual(result.messages.length, 1);
  assert.strictEqual(result.messages[0].content, "hello\nhow are you?");
});
