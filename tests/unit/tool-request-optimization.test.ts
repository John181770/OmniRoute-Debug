import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  isNormalChat,
  compressToolDescription,
  compressSchema,
  deduplicateTools,
  detectRequiredTools,
  mergeSystemPrompts,
  cleanEmptyFields,
  logPayloadStats,
} from "../../open-sse/translator/helpers/toolOptimizer.ts";
import { openaiToClaudeRequest } from "../../open-sse/translator/request/openai-to-claude.ts";
import { claudeToOpenAIRequest } from "../../open-sse/translator/request/claude-to-openai.ts";
import { openaiToGeminiRequest } from "../../open-sse/translator/request/openai-to-gemini.ts";
import { translateOpenAIResponsesToChatCompletions } from "../../open-sse/translator/request/openai-responses.ts";

describe("Tool Request Token Usage Optimizer", () => {
  const sampleTools = [
    {
      type: "function",
      function: {
        name: "web_search",
        description:
          "This tool allows you to search Google for the latest web information, news, and weather details.",
        parameters: {
          $schema: "http://json-schema.org/draft-07/schema#",
          title: "WebSearchParameters",
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "The search query term to submit to Google",
              examples: ["weather in Tokyo"],
            },
          },
          required: ["query"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "image_generation",
        description:
          "Use this tool to generate high quality AI images, illustrations, and photos based on a text prompt.",
        parameters: {
          type: "object",
          properties: {
            prompt: { type: "string", description: "Detailed description of the image to render" },
          },
          required: ["prompt"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "file_reader",
        description:
          "A tool to read PDF files, text documents, and attachments from local storage or URLs.",
        parameters: {
          type: "object",
          properties: {
            path: { type: "string", description: "Path or URL to the document file" },
          },
          required: ["path"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "ocr",
        description:
          "Extract text from image files, scanned PDFs, and screenshots using optical character recognition.",
        parameters: {
          type: "object",
          properties: {
            image_url: { type: "string", description: "URL or base64 string of the image" },
          },
          required: ["image_url"],
        },
      },
    },
  ];

  it("Goal 1 — Normal chat produces 0 tools", () => {
    assert.equal(isNormalChat("hi"), true);
    assert.equal(isNormalChat("hello"), true);
    assert.equal(isNormalChat("good morning"), true);
    assert.equal(isNormalChat("thanks"), true);

    const normalMessages = [{ role: "user", content: "hi" }];
    const required = detectRequiredTools(normalMessages, sampleTools);
    assert.equal(required.length, 0);
  });

  it("Goal 1 — Dynamic tool injection selects specific tool for intent", () => {
    const searchReq = detectRequiredTools(
      [{ role: "user", content: "Search Google for weather" }],
      sampleTools
    );
    assert.equal(searchReq.length, 1);
    assert.equal(searchReq[0].function.name, "web_search");

    const imageReq = detectRequiredTools(
      [{ role: "user", content: "Generate an image of a red cat" }],
      sampleTools
    );
    assert.equal(imageReq.length, 1);
    assert.equal(imageReq[0].function.name, "image_generation");

    const pdfReq = detectRequiredTools(
      [{ role: "user", content: "Read this PDF report" }],
      sampleTools
    );
    assert.equal(pdfReq.length, 1);
    assert.equal(pdfReq[0].function.name, "file_reader");

    const ocrReq = detectRequiredTools(
      [{ role: "user", content: "OCR this image for text" }],
      sampleTools
    );
    assert.ok(ocrReq.length >= 1 && ocrReq.length <= 3);
    assert.ok(
      ocrReq.some(
        (t: Record<string, unknown>) => (t.function as Record<string, unknown>)?.name === "ocr"
      )
    );
  });

  it("Goal 2 — Compress tool descriptions and schemas", () => {
    const rawDesc =
      "This tool allows you to search Google for the latest web information, news, and weather details.";
    const compressedDesc = compressToolDescription(rawDesc);
    assert.ok(!compressedDesc.startsWith("This tool allows you to"));
    assert.ok(compressedDesc.length <= 120);

    const verboseSchema = {
      $schema: "http://json-schema.org/draft-07/schema#",
      title: "MySchema",
      examples: ["ex1"],
      type: "object",
      properties: {
        foo: {
          type: "string",
          description: "Very long description line 1\nVery long description line 2",
        },
      },
      required: ["foo"],
    };
    const clean = compressSchema(verboseSchema);
    assert.equal(clean.$schema, undefined);
    assert.equal(clean.title, undefined);
    assert.equal(clean.examples, undefined);
    assert.equal(clean.type, "object");
    assert.deepEqual(clean.required, ["foo"]);
  });

  it("Goal 3 — Merge multiple system prompts into 1 combined system message", () => {
    const messages = [
      { role: "system", content: "System Prompt A" },
      { role: "system", content: "System Prompt B" },
      { role: "user", content: "Hello" },
    ];
    const merged = mergeSystemPrompts(messages);
    const systemMsgs = merged.filter((m) => m.role === "system");
    assert.equal(systemMsgs.length, 1);
    assert.equal(systemMsgs[0].content, "System Prompt A\n\nSystem Prompt B");
    assert.equal(merged.length, 2);
  });

  it("Goal 4 — Deduplicate duplicate tool definitions", () => {
    const duplicateTools = [sampleTools[0], sampleTools[0], sampleTools[1]];
    const dedupped = deduplicateTools(duplicateTools);
    assert.equal(dedupped.length, 2);
    assert.equal(dedupped[0].function.name, "web_search");
    assert.equal(dedupped[1].function.name, "image_generation");
  });

  it("Goal 5 — Skip empty tool arrays and delete tools / tool_choice", () => {
    const payload: Record<string, unknown> = {
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
      tools: [],
      tool_choice: "auto",
      stop: [],
    };
    cleanEmptyFields(payload);
    assert.equal("tools" in payload, false);
    assert.equal("tool_choice" in payload, false);
    assert.equal("stop" in payload, false);
  });

  it("Goal 6 — Output formatted payload stats log", () => {
    const payload: Record<string, unknown> = {
      model: "gpt-4o",
      messages: [
        { role: "system", content: "System Prompt" },
        { role: "user", content: "hi" },
      ],
    };
    const logs: string[] = [];
    const origLog = console.log;
    console.log = (...args: unknown[]) => {
      logs.push(args.join(" "));
    };
    try {
      logPayloadStats(payload);
    } finally {
      console.log = origLog;
    }

    assert.ok(logs.includes("===== Payload Stats ====="));
    assert.ok(logs.some((l) => l.startsWith("System: 1")));
    assert.ok(logs.some((l) => l.startsWith("Tools: 0")));
    assert.ok(logs.some((l) => l.startsWith("Messages: 2")));
    assert.ok(logs.some((l) => l.startsWith("Payload:")));
    assert.ok(logs.some((l) => l.startsWith("Approx Tokens:")));
  });

  it("Full Translator Test — Normal Chat keeps System: 1, Tools: 0, Token Count low", () => {
    const input = {
      model: "claude-3-7-sonnet",
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "hi" },
      ],
      tools: sampleTools,
    };

    const translated = openaiToClaudeRequest("claude-3-7-sonnet", input, false, {});
    assert.equal(translated.tools, undefined);
    assert.equal(Array.isArray(translated.system), true);
    assert.equal((translated.system as unknown[]).length, 1);

    const payload = JSON.stringify(translated);
    assert.ok(
      payload.length < 1000,
      `Payload length ${payload.length} should be < 1000 for normal chat`
    );
  });

  it("Full Translator Test — Tool request serializes minimum required tools & token budget < 5000", () => {
    const input = {
      model: "claude-3-7-sonnet",
      messages: [
        { role: "system", content: "System Prompt A" },
        { role: "system", content: "System Prompt B" },
        { role: "user", content: "Search Google for latest tech news" },
      ],
      tools: sampleTools,
    };

    const translated = openaiToClaudeRequest("claude-3-7-sonnet", input, false, {});
    assert.ok(Array.isArray(translated.tools));
    assert.equal(translated.tools.length, 1);
    assert.ok(translated.tools[0].name.includes("web_search"));

    const payload = JSON.stringify(translated);
    const approxTokens = Math.ceil(payload.length / 4);
    assert.ok(approxTokens < 5000, `Approx tokens ${approxTokens} should be < 5000`);
  });
});
