import { register } from "../registry.ts";
import { FORMATS } from "../formats.ts";
import { adjustMaxTokens } from "../helpers/maxTokensHelper.ts";
import { coerceSchemaNumericFields } from "../helpers/schemaCoercion.ts";

type JsonRecord = Record<string, unknown>;
const TOOL_CHOICE_ANY = ["a", "n", "y"].join("");

/**
 * Port of decolua/9router commit 0aaa5ab3 (closes prompt-cache instability):
 * Anthropic injects a dynamic `x-anthropic-billing-header: <value>` line at
 * the top of some system prompts. When we translate Claude → OpenAI and
 * forward to a non-Anthropic upstream, that line both leaks into the
 * assistant prompt and rotates per request, destroying prompt-cache hits.
 * Strip it from each system entry before assembling the OpenAI request.
 */
function stripAnthropicBillingHeader(text: unknown): string {
  if (typeof text !== "string") return "";
  return text.replace(/^x-anthropic-billing-header:[^\n]*(?:\r?\n)?/i, "");
}

/**
 * Normalize tool input schema for OpenAI compatibility.
 * OpenAI strict mode requires `properties: {}` on object-type schemas,
 * even for zero-argument tools. Anthropic/MCP tools may omit it (#1898).
 */
function normalizeToolSchema(schema: unknown): Record<string, unknown> {
  const fallback = { type: "object", properties: {} };
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) return fallback;
  const s = schema as Record<string, unknown>;
  if (s.type === "object" && !s.properties) {
    return { ...s, properties: {} };
  }
  return s;
}

function normalizeOpenAIReasoningEffort(effort: unknown): string | undefined {
  if (typeof effort !== "string") return undefined;
  const normalized = effort.toLowerCase();
  if (normalized === "max") return "xhigh";
  return normalized || undefined;
}

function isClaudeServerWebSearchTool(tool: unknown): tool is JsonRecord {
  if (!tool || typeof tool !== "object" || Array.isArray(tool)) return false;
  const record = tool as JsonRecord;
  return (
    record.name === "web_search" &&
    typeof record.type === "string" &&
    /^web_search_\d{8}$/.test(record.type)
  );
}

function toStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((entry): entry is string => typeof entry === "string")
    .map((entry) => entry.trim())
    .filter(Boolean);
}

function convertClaudeServerWebSearchTool(tool: JsonRecord): JsonRecord {
  const allowedDomains = toStringArray(tool.allowed_domains);
  const blockedDomains = toStringArray(tool.blocked_domains);
  const filters: JsonRecord = {};
  if (allowedDomains.length > 0) filters.allowed_domains = allowedDomains;
  if (blockedDomains.length > 0) filters.blocked_domains = blockedDomains;

  return {
    type: "web_search",
    ...(Object.keys(filters).length > 0 ? { filters } : {}),
    ...(tool.user_location &&
    typeof tool.user_location === "object" &&
    !Array.isArray(tool.user_location)
      ? { user_location: tool.user_location }
      : {}),
  };
}

function hasClaudeServerWebSearchTool(tools: unknown): boolean {
  return Array.isArray(tools) && tools.some((tool) => isClaudeServerWebSearchTool(tool));
}

function shouldUseNativeResponsesWebSearch(credentials: unknown): boolean {
  return (
    credentials !== null &&
    typeof credentials === "object" &&
    !Array.isArray(credentials) &&
    (credentials as JsonRecord)._targetFormat === FORMATS.OPENAI_RESPONSES
  );
}

function compressToolDescription(desc: unknown): string {
  if (typeof desc !== "string" || !desc.trim()) return "";
  const firstLine = desc.split("\n")[0].trim();
  const cleaned = firstLine.replace(
    /^(this tool allows you to|use this tool to|a tool to)\s+/i,
    ""
  );
  return cleaned.length > 120 ? cleaned.slice(0, 117) + "..." : cleaned;
}

function isNormalChat(userText: string): boolean {
  if (!userText || !userText.trim()) return true;
  const clean = userText
    .toLowerCase()
    .trim()
    .replace(/[^\w\s]/g, "");

  const simpleGreetings = new Set([
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "thx",
    "how are you",
    "good morning",
    "good afternoon",
    "good evening",
    "who are you",
    "what is your name",
    "what can you do",
    "what is 22",
    "ok",
    "okay",
    "bye",
    "goodbye",
    "yes",
    "no",
    "cool",
    "nice",
    "awesome",
    "great",
  ]);

  if (simpleGreetings.has(clean)) return true;

  const words = clean.split(/\s+/);
  if (words.length <= 2) {
    const greetingWords = new Set([
      "hi",
      "hello",
      "hey",
      "sup",
      "yo",
      "thanks",
      "thank",
      "you",
      "good",
      "morning",
      "afternoon",
      "evening",
      "bye",
      "ok",
    ]);
    if (words.every((w) => greetingWords.has(w))) return true;
  }

  return false;
}

function detectRequiredTools(
  messages: JsonRecord[],
  allTools: JsonRecord[],
  toolChoice: unknown
): JsonRecord[] {
  if (!allTools || allTools.length === 0) return [];
  if (allTools.length === 1 && toolChoice !== "none") return allTools;

  // Check explicit tool_choice override
  if (toolChoice) {
    if (typeof toolChoice === "string") {
      if (toolChoice === "none") return [];
      if (toolChoice === "required" || toolChoice === TOOL_CHOICE_ANY) return allTools;
    } else if (typeof toolChoice === "object" && toolChoice !== null) {
      const choiceObj = toolChoice as JsonRecord;
      let targetName = "";
      if (typeof choiceObj.name === "string") targetName = choiceObj.name;
      else if (choiceObj.function && typeof (choiceObj.function as JsonRecord).name === "string") {
        targetName = (choiceObj.function as JsonRecord).name;
      }
      if (targetName) {
        const matched = allTools.filter((t) => {
          const fn = (t.function as JsonRecord) || t;
          return fn?.name === targetName || t.name === targetName || t.type === targetName;
        });
        if (matched.length > 0) return matched;
      }
    }
  }

  // Gather historical tool call names and message text
  const historicalToolNames = new Set<string>();
  const textParts: string[] = [];

  for (const msg of messages) {
    if (Array.isArray(msg.tool_calls)) {
      for (const tc of msg.tool_calls as JsonRecord[]) {
        const fn = tc.function as JsonRecord | undefined;
        if (fn?.name && typeof fn.name === "string") {
          historicalToolNames.add(fn.name);
        }
      }
    }
    if (typeof msg.content === "string") {
      textParts.push(msg.content);
    } else if (Array.isArray(msg.content)) {
      for (const part of msg.content as JsonRecord[]) {
        if (part?.type === "text" && typeof part.text === "string") {
          textParts.push(part.text);
        }
      }
    }
  }

  const combinedText = textParts.join(" ").toLowerCase();
  const latestUserMsg = [...messages].reverse().find((m) => m.role === "user");
  let latestUserText = "";
  if (latestUserMsg) {
    if (typeof latestUserMsg.content === "string") {
      latestUserText = latestUserMsg.content;
    } else if (Array.isArray(latestUserMsg.content)) {
      latestUserText = (latestUserMsg.content as JsonRecord[])
        .filter((p) => p?.type === "text" && typeof p.text === "string")
        .map((p) => p.text)
        .join(" ");
    }
  }

  // Task 3: Skip tools for normal chat if no historical tool calls are pending
  if (isNormalChat(latestUserText) && historicalToolNames.size === 0) {
    return [];
  }

  const required: JsonRecord[] = [];

  for (const tool of allTools) {
    const fn = (tool.function as JsonRecord) || tool;
    const name =
      typeof fn.name === "string"
        ? fn.name.toLowerCase()
        : typeof tool.type === "string"
          ? tool.type.toLowerCase()
          : "";
    const desc = typeof fn.description === "string" ? fn.description.toLowerCase() : "";

    // 1. Tool used in historical conversation turn
    if (name && historicalToolNames.has(name)) {
      required.push(tool);
      continue;
    }

    // 2. Direct tool name match or sub-token match
    if (
      name &&
      (combinedText.includes(name) ||
        name.split("_").every((k) => k.length > 2 && combinedText.includes(k)))
    ) {
      required.push(tool);
      continue;
    }

    // 3. Domain / Intent match
    let isMatch = false;

    if (name.includes("search") || name.includes("google") || name === "web_search") {
      if (
        /\b(search|google|lookup|find online|browse|web|website|url|http|https|news|latest|weather)\b/i.test(
          combinedText
        )
      ) {
        isMatch = true;
      }
    } else if (name.includes("image") || name.includes("draw") || name.includes("dall")) {
      if (
        /\b(generate image|create image|draw|picture|photo|illustration|render image|image)\b/i.test(
          combinedText
        )
      ) {
        isMatch = true;
      }
    } else if (name.includes("file") || name.includes("pdf") || name.includes("read")) {
      if (/\b(file|pdf|read file|read pdf|document|attachment|open file)\b/i.test(combinedText)) {
        isMatch = true;
      }
    } else if (
      name.includes("code") ||
      name.includes("python") ||
      name.includes("bash") ||
      name.includes("terminal") ||
      name.includes("exec")
    ) {
      if (/\b(code|python|bash|terminal|script|execute|run)\b/i.test(combinedText)) {
        isMatch = true;
      }
    } else if (desc) {
      const keywords = name.split(/[\s_-]+/).filter((k) => k.length > 3);
      if (keywords.some((k) => combinedText.includes(k))) {
        isMatch = true;
      }
    }

    if (isMatch) {
      required.push(tool);
    }
  }

  return required;
}

// Convert Claude request to OpenAI format
export function claudeToOpenAIRequest(model, body, stream, credentials: unknown = null) {
  // #2069 — when the routed provider honors OpenAI-format cache_control breakpoints
  // (DashScope/alibaba, etc.) and the upstream caller requested preservation, keep
  // the client's cache_control markers on system + message text blocks instead of
  // collapsing them away during the Claude→OpenAI conversion.
  const preserveCacheControl =
    credentials !== null &&
    typeof credentials === "object" &&
    !Array.isArray(credentials) &&
    (credentials as JsonRecord)._preserveCacheControl === true;

  const result: {
    model: string;
    messages: JsonRecord[];
    stream: unknown;
    [key: string]: unknown;
  } = {
    model: model,
    messages: [],
    stream: stream,
  };

  // Max tokens
  if (body.max_tokens) {
    result.max_tokens = adjustMaxTokens(body);
  }

  // Temperature
  if (body.temperature !== undefined) {
    result.temperature = body.temperature;
  }
  if (body.top_p !== undefined) {
    result.top_p = body.top_p;
  }
  if (body.stop_sequences !== undefined) {
    result.stop = body.stop_sequences;
  }

  // Task 4: Consolidate & Merge System Prompts (from body.system and body.messages)
  const systemBlocks: unknown[] = [];
  if (body.system) {
    if (Array.isArray(body.system)) systemBlocks.push(...body.system);
    else systemBlocks.push(body.system);
  }

  // Extract mid-conversation system messages from body.messages
  const nonSystemBodyMessages: JsonRecord[] = [];
  if (body.messages && Array.isArray(body.messages)) {
    for (const msg of body.messages) {
      if (msg && typeof msg === "object" && msg.role === "system") {
        if (typeof msg.content === "string") systemBlocks.push(msg.content);
        else if (Array.isArray(msg.content)) systemBlocks.push(...msg.content);
      } else {
        nonSystemBodyMessages.push(msg);
      }
    }
  }

  if (systemBlocks.length > 0) {
    const systemHasCacheControl =
      preserveCacheControl &&
      systemBlocks.some(
        (s) => s && typeof s === "object" && (s as JsonRecord).cache_control !== undefined
      );

    let systemContent: unknown;
    if (systemHasCacheControl) {
      const seenTexts = new Set<string>();
      systemContent = systemBlocks
        .map((s) => {
          if (typeof s === "string") {
            const text = stripAnthropicBillingHeader(s).trim();
            if (!text || seenTexts.has(text)) return null;
            seenTexts.add(text);
            return { type: "text", text };
          }
          const rawText = s && typeof s === "object" ? (s as JsonRecord).text || "" : "";
          const text = stripAnthropicBillingHeader(rawText).trim();
          if (!text || seenTexts.has(text)) return null;
          seenTexts.add(text);
          const block: JsonRecord = { type: "text", text };
          if (s && typeof s === "object" && (s as JsonRecord).cache_control !== undefined) {
            block.cache_control = (s as JsonRecord).cache_control;
          }
          return block;
        })
        .filter(Boolean);
    } else {
      const uniqueLines: string[] = [];
      const seen = new Set<string>();
      for (const s of systemBlocks) {
        const raw = typeof s === "string" ? s : s?.text || "";
        const cleaned = stripAnthropicBillingHeader(raw).trim();
        if (!cleaned) continue;
        const lines = cleaned.split("\n");
        for (const line of lines) {
          const trimmedLine = line.trim();
          if (trimmedLine && !seen.has(trimmedLine)) {
            seen.add(trimmedLine);
            uniqueLines.push(trimmedLine);
          }
        }
      }
      systemContent = uniqueLines.join("\n");
    }

    if (
      systemHasCacheControl ||
      (typeof systemContent === "string"
        ? Boolean(systemContent)
        : Array.isArray(systemContent) && systemContent.length > 0)
    ) {
      result.messages.push({
        role: "system",
        content: systemContent,
      });
    }
  }

  // Convert messages
  for (let i = 0; i < nonSystemBodyMessages.length; i++) {
    const msg = nonSystemBodyMessages[i];
    const converted = convertClaudeMessage(msg, preserveCacheControl);
    if (converted) {
      if (Array.isArray(converted)) {
        result.messages.push(...converted);
      } else {
        result.messages.push(converted);
      }
    }
  }

  // Deduplicate images in historical turns (preserves full image on current/latest turn)
  result.messages = deduplicateAndOptimizeImages(result.messages);

  // #4714 / #4385: re-group every role:"tool" message immediately after the
  // assistant turn whose tool_calls issued it (dropping genuine orphans), then
  // fill placeholders for any tool_call left unanswered.
  result.messages = regroupToolMessages(result.messages);

  // Merge adjacent turns of same role & remove empty content turns
  result.messages = mergeAdjacentAndRemoveEmpty(result.messages);

  // Fix missing tool responses - OpenAI requires every tool_call to have a response.
  // Runs after regrouping so real results are already adjacent and only a truly
  // unanswered tool_call receives a placeholder.
  fixMissingToolResponses(result.messages);

  const useNativeResponsesWebSearch = shouldUseNativeResponsesWebSearch(credentials);

  // Tools & Schema Compression (Task 2)
  let normalizedTools: JsonRecord[] = [];
  if (body.tools && Array.isArray(body.tools)) {
    const seenToolNames = new Set<string>();
    normalizedTools = body.tools
      .map((tool) => {
        if (isClaudeServerWebSearchTool(tool)) {
          if (useNativeResponsesWebSearch) {
            return convertClaudeServerWebSearchTool(tool);
          }
          return {
            type: "function",
            function: {
              name: "web_search",
              description: "",
              parameters: { type: "object", properties: {}, additionalProperties: true },
            },
          };
        }

        if (!tool || typeof tool !== "object" || Array.isArray(tool)) return null;
        const record = tool as JsonRecord;
        const name = typeof record.name === "string" ? record.name.trim() : "";
        if (!name || seenToolNames.has(name)) return null; // skip duplicate or empty name
        seenToolNames.add(name);

        return {
          type: "function",
          function: {
            name,
            description: compressToolDescription(record.description),
            parameters: coerceSchemaNumericFields(normalizeToolSchema(record.input_schema)),
          },
        };
      })
      .filter((tool): tool is JsonRecord => Boolean(tool));
  }

  // Dynamic Tool Injection (Task 1 & Task 3)
  const requiredTools = detectRequiredTools(result.messages, normalizedTools, body.tool_choice);
  if (requiredTools.length > 0) {
    result.tools = requiredTools;
  }

  // Tool choice
  if (body.tool_choice && result.tools && Array.isArray(result.tools) && result.tools.length > 0) {
    result.tool_choice = convertToolChoice(
      body.tool_choice,
      useNativeResponsesWebSearch && hasClaudeServerWebSearchTool(body.tools)
    );
  }

  // Reasoning effort: map Claude-side thinking controls to OpenAI reasoning_effort.
  // Priority: output_config.effort (Claude Code) > thinking.budget_tokens (Claude native).
  // Budget buckets match the reverse mapping in thinkingBudget.ts::setCustomBudget.
  const outputEffort = normalizeOpenAIReasoningEffort(body.output_config?.effort) || "";
  if (outputEffort) {
    result.reasoning_effort = outputEffort;
  } else if (body.thinking?.type === "enabled" && typeof body.thinking.budget_tokens === "number") {
    const budget = body.thinking.budget_tokens;
    if (budget <= 0) {
      // disabled — leave reasoning_effort unset
    } else if (budget <= 1024) {
      result.reasoning_effort = "low";
    } else if (budget <= 10240) {
      result.reasoning_effort = "medium";
    } else if (budget < 131072) {
      result.reasoning_effort = "high";
    } else {
      result.reasoning_effort = "xhigh";
    }
  }

  // Remove Empty Request Fields (Task 8 & Task 9)
  if (!result.tools || (Array.isArray(result.tools) && result.tools.length === 0)) {
    delete result.tools;
    delete result.tool_choice;
  }
  if (!result.tool_choice) {
    delete result.tool_choice;
  }
  if (!result.stop || (Array.isArray(result.stop) && result.stop.length === 0)) {
    delete result.stop;
  }

  const payload = JSON.stringify(result);

  // Instrumentation (Task 10)
  const systemCount = result.messages.filter((m) => m.role === "system").length;
  const toolCount = Array.isArray(result.tools) ? (result.tools as unknown[]).length : 0;

  console.log("\n===== Payload Stats =====");
  console.log("System:", systemCount);
  console.log("Tools:", toolCount);
  console.log("Messages:", result.messages.length);
  console.log("Payload:", payload.length);
  console.log("Approx Tokens:", Math.ceil(payload.length / 4));
  console.log("================================\n");

  return result;
}

function getImageFingerprint(url: string): string {
  if (!url || typeof url !== "string") return "";
  const commaIdx = url.indexOf(",");
  if (commaIdx !== -1 && url.startsWith("data:image/")) {
    const b64 = url.slice(commaIdx + 1).trim();
    if (b64.length > 100) {
      return `b64_${b64.length}_${b64.slice(0, 32)}_${b64.slice(-32)}`;
    }
  }
  return url;
}

function deduplicateAndOptimizeImages(messages: JsonRecord[]): JsonRecord[] {
  const lastIndexMap = new Map<string, number>();

  messages.forEach((msg, idx) => {
    if (Array.isArray(msg.content)) {
      for (const part of msg.content as JsonRecord[]) {
        if (part?.type === "image_url" && part.image_url && typeof part.image_url === "object") {
          const url = (part.image_url as JsonRecord).url;
          if (typeof url === "string" && url.length > 100) {
            const fp = getImageFingerprint(url);
            lastIndexMap.set(fp, idx);
          }
        }
      }
    }
  });

  return messages.map((msg, idx) => {
    if (!Array.isArray(msg.content)) return msg;
    const newContent: JsonRecord[] = [];
    const seenFingerprintsInTurn = new Set<string>();

    for (const part of msg.content as JsonRecord[]) {
      if (part?.type === "image_url" && part.image_url && typeof part.image_url === "object") {
        const url = (part.image_url as JsonRecord).url;
        if (typeof url === "string" && url.length > 100) {
          const fp = getImageFingerprint(url);
          const lastIdx = lastIndexMap.get(fp);
          if ((lastIdx !== undefined && lastIdx > idx) || idx < messages.length - 1) {
            newContent.push({
              type: "text",
              text: "[Prior attached image retained in context]",
            });
            continue;
          }
          if (seenFingerprintsInTurn.has(fp)) {
            newContent.push({
              type: "text",
              text: "[Duplicate attached image retained in context]",
            });
            continue;
          }
          seenFingerprintsInTurn.add(fp);
        }
      }
      newContent.push(part);
    }

    const collapsedContent: JsonRecord[] = [];
    for (const item of newContent) {
      if (
        item.type === "text" &&
        collapsedContent.length > 0 &&
        collapsedContent[collapsedContent.length - 1].type === "text"
      ) {
        collapsedContent[collapsedContent.length - 1].text += "\n" + item.text;
      } else {
        collapsedContent.push(item);
      }
    }

    return {
      ...msg,
      content:
        collapsedContent.length === 1 && collapsedContent[0].type === "text"
          ? collapsedContent[0].text
          : collapsedContent,
    };
  });
}

function mergeAdjacentAndRemoveEmpty(messages: JsonRecord[]): JsonRecord[] {
  const filtered: JsonRecord[] = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = { ...messages[i] };
    if (typeof msg.content === "string") {
      msg.content = msg.content.replace(/\n{3,}/g, "\n\n");
    }
    const isFinalAssistant = i === messages.length - 1 && msg.role === "assistant";
    const hasReasoning = "reasoning_content" in msg && msg.reasoning_content !== null;
    if (
      !isFinalAssistant &&
      (msg.role === "user" || msg.role === "assistant") &&
      !msg.tool_calls &&
      !hasReasoning &&
      !msg.tool_call_id
    ) {
      if (typeof msg.content === "string" && msg.content.trim() === "") continue;
      if (Array.isArray(msg.content) && msg.content.length === 0) continue;
    }
    filtered.push(msg);
  }

  const merged: JsonRecord[] = [];
  for (const msg of filtered) {
    const prev = merged[merged.length - 1];

    // Task 6: Remove duplicate adjacent messages with identical role and content
    const prevReasoning = prev && "reasoning_content" in prev ? prev.reasoning_content : undefined;
    const msgReasoning = "reasoning_content" in msg ? msg.reasoning_content : undefined;

    if (
      prev &&
      prev.role === msg.role &&
      !prev.tool_calls &&
      !msg.tool_calls &&
      !prev.tool_call_id &&
      !msg.tool_call_id &&
      prevReasoning === msgReasoning &&
      JSON.stringify(prev.content) === JSON.stringify(msg.content)
    ) {
      continue;
    }

    if (
      prev &&
      prev.role === "user" &&
      msg.role === "user" &&
      !prev.tool_calls &&
      !msg.tool_calls &&
      !prev.tool_call_id &&
      !msg.tool_call_id
    ) {
      const prevContent =
        typeof prev.content === "string"
          ? [{ type: "text", text: prev.content }]
          : Array.isArray(prev.content)
            ? prev.content
            : [];
      const msgContent =
        typeof msg.content === "string"
          ? [{ type: "text", text: msg.content }]
          : Array.isArray(msg.content)
            ? msg.content
            : [];
      const combined = [...prevContent, ...msgContent];

      if (combined.every((p) => p.type === "text")) {
        prev.content = combined.map((p) => p.text).join("\n");
      } else {
        prev.content = combined;
      }
      continue;
    }

    const prevHasReasoning = prev && "reasoning_content" in prev && prev.reasoning_content !== null;
    const msgHasReasoning = "reasoning_content" in msg && msg.reasoning_content !== null;

    if (
      prev &&
      prev.role === "assistant" &&
      msg.role === "assistant" &&
      !prev.tool_calls &&
      !msg.tool_calls &&
      !prevHasReasoning &&
      !msgHasReasoning
    ) {
      const prevContent =
        typeof prev.content === "string"
          ? prev.content
          : Array.isArray(prev.content)
            ? (prev.content as JsonRecord[]).map((p) => p.text || "").join("\n")
            : "";
      const msgContent =
        typeof msg.content === "string"
          ? msg.content
          : Array.isArray(msg.content)
            ? (msg.content as JsonRecord[]).map((p) => p.text || "").join("\n")
            : "";
      prev.content = [prevContent, msgContent].filter(Boolean).join("\n");
      continue;
    }

    merged.push(msg);
  }

  return merged;
}

// #4714: Re-group tool result messages so every role:"tool" message sits
// immediately after the assistant message whose tool_calls issued it, in
// tool_calls order. Claude Code can issue parallel tool_use in one assistant turn
// but have their tool_result blocks arrive across SEPARATE user turns with
// interleaved text; the naive conversion then left a role:"tool" stranded after a
// user message, which OpenAI-compatible upstreams reject with 400 "Messages with
// role 'tool' must be a response to a preceding message with 'tool_calls'".
// Tool messages whose tool_call_id matches no assistant.tool_calls are dropped here
// (supersedes the standalone #4385 orphan filter — same drop behavior, plus ordering).
function regroupToolMessages(messages: JsonRecord[]): JsonRecord[] {
  // tool_call_id -> index of the assistant message that issued it (first wins).
  const callIdToAssistant = new Map<string, number>();
  messages.forEach((msg, idx) => {
    if (msg.role === "assistant" && Array.isArray(msg.tool_calls)) {
      for (const tc of msg.tool_calls as { id?: string }[]) {
        if (tc.id && !callIdToAssistant.has(String(tc.id))) {
          callIdToAssistant.set(String(tc.id), idx);
        }
      }
    }
  });

  // Collect tool messages per assistant index, keyed by tool_call_id (first result wins).
  const toolsByAssistant = new Map<number, Map<string, JsonRecord>>();
  for (const msg of messages) {
    if (msg.role !== "tool") continue;
    const callId = String(msg.tool_call_id ?? "");
    const assistantIdx = callIdToAssistant.get(callId);
    if (assistantIdx === undefined) continue; // orphan -> drop
    let group = toolsByAssistant.get(assistantIdx);
    if (!group) {
      group = new Map();
      toolsByAssistant.set(assistantIdx, group);
    }
    if (!group.has(callId)) group.set(callId, msg);
  }

  // Rebuild: keep non-tool messages in order; attach each assistant's tool results
  // immediately after it, ordered by the assistant's own tool_calls sequence.
  const out: JsonRecord[] = [];
  messages.forEach((msg, idx) => {
    if (msg.role === "tool") return; // moved into its assistant's group
    out.push(msg);
    if (msg.role === "assistant" && Array.isArray(msg.tool_calls)) {
      const group = toolsByAssistant.get(idx);
      if (group) {
        for (const tc of msg.tool_calls as { id?: string }[]) {
          const tool = tc.id ? group.get(String(tc.id)) : undefined;
          if (tool) out.push(tool);
        }
      }
    }
  });
  return out;
}

// Fix missing tool responses - add empty responses for tool_calls without responses
function fixMissingToolResponses(messages) {
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (msg.role === "assistant" && msg.tool_calls && msg.tool_calls.length > 0) {
      const toolCallIds = msg.tool_calls.map((tc) => tc.id);

      // Collect all tool response IDs that IMMEDIATELY follow this assistant message
      const respondedIds = new Set();
      let insertPosition = i + 1;
      for (let j = i + 1; j < messages.length; j++) {
        const nextMsg = messages[j];
        if (nextMsg.role === "tool" && nextMsg.tool_call_id) {
          respondedIds.add(nextMsg.tool_call_id);
          insertPosition = j + 1;
        } else {
          break;
        }
      }

      // Find missing responses and insert them
      const missingIds = toolCallIds.filter((id) => !respondedIds.has(id));

      if (missingIds.length > 0) {
        const missingResponses = missingIds.map((id) => ({
          role: "tool",
          tool_call_id: id,
          content: "[No response received]",
        }));
        messages.splice(insertPosition, 0, ...missingResponses);
        i = insertPosition + missingResponses.length - 1;
      }
    }
  }
}

// Convert single Claude message - returns single message or array of messages
function convertClaudeMessage(msg, preserveCacheControl = false) {
  // Preserve system role for mid-conversation system turns (#6954).
  // Previously any role that wasn't "user" or "tool" was mapped to "assistant",
  // which misattributed system messages as assistant output.
  const role =
    msg.role === "user" || msg.role === "tool"
      ? "user"
      : msg.role === "system"
        ? "system"
        : "assistant";

  // Simple string content
  if (typeof msg.content === "string") {
    return { role, content: msg.content };
  }

  // Array content
  if (Array.isArray(msg.content)) {
    const parts = [];
    const toolCalls = [];
    const toolResults = [];
    let reasoningContent = null;

    for (const block of msg.content) {
      switch (block.type) {
        case "text": {
          const textPart: JsonRecord = { type: "text", text: block.text };
          // #2069 — carry the client's cache_control breakpoint through to
          // caching-capable OpenAI-format providers (DashScope/alibaba, etc.).
          if (preserveCacheControl && block.cache_control !== undefined) {
            textPart.cache_control = block.cache_control;
          }
          parts.push(textPart);
          break;
        }

        case "image":
          if (block.source?.type === "base64") {
            parts.push({
              type: "image_url",
              image_url: {
                url: `data:${block.source.media_type};base64,${block.source.data}`,
              },
            });
          } else if (block.source?.type === "url" && typeof block.source.url === "string") {
            parts.push({
              type: "image_url",
              image_url: {
                url: block.source.url,
              },
            });
          }
          break;

        case "thinking":
          reasoningContent = block.thinking || block.text || "";
          break;

        case "redacted_thinking":
          if (reasoningContent == null) {
            reasoningContent = "";
          }
          break;

        case "tool_use":
          toolCalls.push({
            id: block.id,
            type: "function",
            function: {
              name: block.name,
              arguments:
                typeof block.input === "string" ? block.input : JSON.stringify(block.input || {}),
            },
          });
          break;

        case "tool_result":
          let resultContent = "";
          if (typeof block.content === "string") {
            resultContent = block.content;
          } else if (Array.isArray(block.content)) {
            // Keep text in the tool message; lift any images out as a following user
            // turn (OpenAI `tool` messages can't carry images). Without this, an
            // image-only tool_result is JSON.stringify'd → base64 as text, which
            // causes "input exceeds the context window" errors in OpenAI-protocol
            // upstreams (port of decolua/9router#2123 by alican532).
            const textParts: string[] = [];
            let hasImage = false;
            for (const c of block.content) {
              if (c.type === "text") {
                textParts.push(c.text);
              } else if (c.type === "image" && c.source?.type === "base64") {
                parts.push({
                  type: "image_url",
                  image_url: {
                    url: `data:${c.source.media_type};base64,${c.source.data}`,
                  },
                });
                hasImage = true;
              }
            }
            resultContent =
              textParts.join("\n") ||
              (hasImage ? "[tool returned an image; see attached]" : JSON.stringify(block.content));
          } else if (block.content) {
            resultContent = JSON.stringify(block.content);
          }

          toolResults.push({
            role: "tool",
            tool_call_id: block.tool_use_id,
            content: resultContent,
          });
          break;
      }
    }

    // If has tool results, return array of tool messages
    if (toolResults.length > 0) {
      if (parts.length > 0) {
        const textContent = parts.length === 1 && parts[0].type === "text" ? parts[0].text : parts;
        return [...toolResults, { role: "user", content: textContent }];
      }
      return toolResults;
    }

    // If has tool calls, return assistant message with tool_calls
    if (toolCalls.length > 0) {
      const result: JsonRecord = { role: "assistant" };
      if (parts.length > 0) {
        result.content = parts.length === 1 && parts[0].type === "text" ? parts[0].text : parts;
      }
      result.tool_calls = toolCalls;
      if (reasoningContent !== null) {
        result.reasoning_content = reasoningContent;
      }
      return result;
    }

    // Return content
    if (parts.length > 0) {
      const result: JsonRecord = {
        role,
        content: parts.length === 1 && parts[0].type === "text" ? parts[0].text : parts,
      };
      if (reasoningContent !== null && role === "assistant") {
        result.reasoning_content = reasoningContent;
      }
      return result;
    }

    // Empty content array
    if (msg.content.length === 0) {
      const result: JsonRecord = { role, content: "" };
      if (reasoningContent !== null && role === "assistant") {
        result.reasoning_content = reasoningContent;
      }
      return result;
    }

    if (reasoningContent !== null && role === "assistant") {
      return { role, content: "", reasoning_content: reasoningContent };
    }
  }

  return null;
}

// Convert tool choice
function convertToolChoice(choice, hasServerWebSearch = false) {
  if (!choice) return "auto";
  if (typeof choice === "string") return choice;

  switch (choice.type) {
    case "auto":
      return "auto";
    case TOOL_CHOICE_ANY:
      return "required";
    case "tool":
      if (hasServerWebSearch && choice.name === "web_search") {
        return { type: "web_search" };
      }
      return { type: "function", function: { name: choice.name } };
    default:
      return "auto";
  }
}

// Register
register(FORMATS.CLAUDE, FORMATS.OPENAI, claudeToOpenAIRequest, null);
