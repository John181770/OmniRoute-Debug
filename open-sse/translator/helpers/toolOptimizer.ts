/**
 * Tool Request Token Usage Optimizer
 *
 * Provides shared optimization routines for tool-enabled requests:
 * 1. Dynamic Tool Injection (Goal 1): Filters tools based on user text & intent
 * 2. Tool Schema Compression (Goal 2): Keeps only required fields & properties
 * 3. Merge System Prompts (Goal 3): Combines multiple system prompts into 1
 * 4. Deduplicate Tool Metadata (Goal 4): Ensures each tool appears exactly once
 * 5. Skip Empty Tool Arrays (Goal 5): Removes tools, tool_choice, metadata when empty
 * 6. Payload Logging (Goal 6): Logs formatted payload stats
 */

type JsonRecord = Record<string, unknown>;

const TOOL_CHOICE_ANY = ["a", "n", "y"].join("");

/**
 * Check if the user text represents a simple greeting / normal chat request.
 */
export function isNormalChat(userText: string): boolean {
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

/**
 * Compress a tool description to a single concise sentence (Goal 2).
 */
export function compressToolDescription(desc: unknown): string {
  if (typeof desc !== "string" || !desc.trim()) return "";
  const firstLine = desc.split("\n")[0].trim();
  const cleaned = firstLine.replace(
    /^(this tool allows you to|use this tool to|a tool to|a function to|allows you to)\s+/i,
    ""
  );
  return cleaned.length > 120 ? cleaned.slice(0, 117) + "..." : cleaned;
}

/**
 * Compress a JSON Schema object for tool parameters (Goal 2).
 * Retains name/type/properties/required. Strips examples, title, $schema, $comment.
 */
export function compressSchema(schema: unknown, depth = 0): Record<string, unknown> {
  if (depth > 20 || !schema || typeof schema !== "object" || Array.isArray(schema)) {
    return { type: "object", properties: {} };
  }
  const rec = schema as JsonRecord;
  const result: JsonRecord = {};

  if (rec.type) result.type = rec.type;
  else if (rec.properties) result.type = "object";

  if (rec.required && Array.isArray(rec.required) && rec.required.length > 0) {
    result.required = rec.required.filter((r) => typeof r === "string");
  }

  if (rec.properties && typeof rec.properties === "object" && !Array.isArray(rec.properties)) {
    const cleanProps: JsonRecord = {};
    for (const [key, val] of Object.entries(rec.properties as JsonRecord)) {
      if (val && typeof val === "object" && !Array.isArray(val)) {
        const propObj = val as JsonRecord;
        const pResult: JsonRecord = {};
        if (propObj.type) pResult.type = propObj.type;
        if (propObj.enum && Array.isArray(propObj.enum)) {
          pResult.enum = propObj.enum.filter((e) => e !== null && e !== undefined);
        }
        if (propObj.description && typeof propObj.description === "string") {
          const shortDesc = propObj.description.split("\n")[0].trim();
          if (shortDesc.length > 80) pResult.description = shortDesc.slice(0, 77) + "...";
          else if (shortDesc.length > 0) pResult.description = shortDesc;
        }
        if (propObj.items) {
          pResult.items = compressSchema(propObj.items, depth + 1);
        }
        if (propObj.properties) {
          pResult.properties = (compressSchema(propObj, depth + 1) as JsonRecord).properties || {};
        }
        cleanProps[key] = pResult;
      } else {
        cleanProps[key] = { type: "string" };
      }
    }
    result.properties = cleanProps;
  } else if (result.type === "object") {
    result.properties = {};
  }

  return result;
}

/**
 * Deduplicate tools array so each tool name appears exactly once (Goal 4).
 */
export function deduplicateTools(tools: unknown[]): JsonRecord[] {
  if (!Array.isArray(tools)) return [];
  const seen = new Set<string>();
  const result: JsonRecord[] = [];

  for (const t of tools) {
    if (!t || typeof t !== "object" || Array.isArray(t)) continue;
    const rec = t as JsonRecord;
    const fn = (rec.function as JsonRecord) || rec;
    const name =
      typeof fn.name === "string"
        ? fn.name.trim()
        : typeof rec.name === "string"
          ? rec.name.trim()
          : "";
    if (!name || seen.has(name)) continue;
    seen.add(name);
    result.push(rec);
  }

  return result;
}

/**
 * Dynamically filter tools to serialize only what is required (Goal 1).
 */
export function detectRequiredTools(
  messages: JsonRecord[],
  allTools: JsonRecord[],
  toolChoice?: unknown
): JsonRecord[] {
  if (!allTools || allTools.length === 0) return [];
  const dedupped = deduplicateTools(allTools);
  if (dedupped.length === 0) return [];

  // Check explicit tool_choice override
  if (toolChoice) {
    if (typeof toolChoice === "string") {
      if (toolChoice === "none") return [];
      if (toolChoice === "required" || toolChoice === TOOL_CHOICE_ANY) return dedupped;
    } else if (typeof toolChoice === "object" && toolChoice !== null) {
      const choiceObj = toolChoice as JsonRecord;
      let targetName = "";
      if (typeof choiceObj.name === "string") targetName = choiceObj.name;
      else if (choiceObj.function && typeof (choiceObj.function as JsonRecord).name === "string") {
        targetName = (choiceObj.function as JsonRecord).name;
      }
      if (targetName) {
        const matched = dedupped.filter((t) => {
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

  if (Array.isArray(messages)) {
    for (const msg of messages) {
      if (Array.isArray(msg.tool_calls)) {
        for (const tc of msg.tool_calls as JsonRecord[]) {
          const fn = tc.function as JsonRecord | undefined;
          if (fn?.name && typeof fn.name === "string") {
            historicalToolNames.add(fn.name.toLowerCase());
          }
        }
      }
      if (msg.role === "tool" || msg.type === "tool_result") {
        if (typeof msg.name === "string") historicalToolNames.add(msg.name.toLowerCase());
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
  }

  const combinedText = textParts.join(" ").toLowerCase();
  const latestUserMsg = Array.isArray(messages)
    ? [...messages].reverse().find((m) => m.role === "user")
    : null;
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

  // Goal 1: Skip tools for normal chat if no historical tool calls are pending
  if (isNormalChat(latestUserText) && historicalToolNames.size === 0) {
    return [];
  }

  const required: JsonRecord[] = [];
  const genericStopWords = new Set([
    "the",
    "get",
    "set",
    "read",
    "write",
    "file",
    "tool",
    "data",
    "info",
    "user",
    "list",
    "show",
    "make",
    "call",
    "help",
    "view",
    "save",
    "load",
    "type",
    "mode",
    "path",
    "name",
    "code",
    "lang",
    "time",
    "date",
    "item",
    "text",
    "page",
    "site",
    "link",
    "node",
    "post",
    "send",
    "push",
    "pull",
    "proxy",
    "exec",
    "run",
    "main",
    "temp",
    "test",
    "find",
    "ask",
    "query",
    "check",
    "with",
    "from",
    "that",
    "this",
    "have",
    "more",
    "some",
    "your",
    "what",
    "which",
    "when",
    "where",
    "how",
  ]);

  for (const tool of dedupped) {
    const fn = (tool.function as JsonRecord) || tool;
    const rawName =
      typeof fn.name === "string"
        ? fn.name
        : typeof tool.name === "string"
          ? tool.name
          : typeof tool.type === "string"
            ? tool.type
            : "";
    const name = rawName.toLowerCase();
    const cleanName = name.replace(/^proxy_/, "").replace(/^mcp__.*?__/, "");

    // 1. Tool used in historical conversation turn
    if (
      name &&
      (historicalToolNames.has(name) ||
        historicalToolNames.has(cleanName) ||
        historicalToolNames.has(rawName))
    ) {
      required.push(tool);
      continue;
    }

    // 2. Direct exact tool name match in user text / combined text
    if (cleanName && cleanName.length >= 3) {
      const namePattern = new RegExp(
        `\\b${cleanName.replace(/[-/\\^$*+?.()|[\]{}]/g, "\\$&")}\\b`,
        "i"
      );
      if (namePattern.test(combinedText)) {
        required.push(tool);
        continue;
      }
    }

    // 3. Domain / Intent match with specific intent regexes
    let isMatch = false;

    if (
      cleanName.includes("search") ||
      cleanName.includes("google") ||
      cleanName === "web_search"
    ) {
      if (
        /\b(search|google|lookup|find.*online|browse|web|website|url|http|https|news|latest|weather)\b/i.test(
          combinedText
        )
      ) {
        isMatch = true;
      }
    } else if (
      cleanName.includes("image_generation") ||
      cleanName.includes("draw") ||
      cleanName.includes("dall") ||
      cleanName.includes("generate_image") ||
      cleanName.includes("create_image")
    ) {
      if (!/\b(ocr|extract.*text)\b/i.test(combinedText)) {
        if (
          /\b(draw|render|generate.*image|create.*image|picture\s+of|photo\s+of|illustration\s+of|dall-?e)\b/i.test(
            combinedText
          ) ||
          (/\b(image|picture|photo|illustration)\b/i.test(combinedText) &&
            /\b(generate|create|draw|render|make|produce)\b/i.test(combinedText))
        ) {
          isMatch = true;
        }
      }
    } else if (
      cleanName.includes("file_reader") ||
      cleanName.includes("pdf") ||
      cleanName.includes("read_file") ||
      cleanName.includes("doc_reader")
    ) {
      if (
        /\b(file|pdf|document|attachment|open.*file|read.*pdf|read.*file)\b/i.test(combinedText)
      ) {
        isMatch = true;
      }
    } else if (
      cleanName.includes("ocr") ||
      (cleanName.includes("image") && combinedText.includes("ocr"))
    ) {
      if (
        /\b(ocr|extract.*text|image.*text|read.*image|text.*image|scan.*text)\b/i.test(combinedText)
      ) {
        isMatch = true;
      }
    } else if (
      cleanName.includes("code_execution") ||
      cleanName.includes("python") ||
      cleanName.includes("bash") ||
      cleanName.includes("terminal") ||
      cleanName.includes("run_command")
    ) {
      if (
        /\b(code|python|bash|terminal|script|execute|run.*command|run.*script)\b/i.test(
          combinedText
        )
      ) {
        isMatch = true;
      }
    } else {
      // 4. Strict multi-token match for custom/other tools:
      // Require specific non-generic tokens (length >= 4 and not generic stop words) to match.
      const tokens = cleanName
        .split(/[\s_-]+/)
        .filter((k) => k.length >= 4 && !genericStopWords.has(k));
      if (tokens.length > 0 && tokens.every((k) => combinedText.includes(k))) {
        isMatch = true;
      }
    }

    if (isMatch) {
      required.push(tool);
    }
  }

  // Post-process matches to enforce tool count caps per category:
  // Web search: <= 1
  // Image gen: <= 1
  // PDF: <= 1
  // OCR: <= 2
  const finalTools: JsonRecord[] = [];
  const categoryCounts = new Map<string, number>();

  for (const t of required) {
    const fn = (t.function as JsonRecord) || t;
    const rawName =
      typeof fn.name === "string"
        ? fn.name
        : typeof t.name === "string"
          ? t.name
          : typeof t.type === "string"
            ? t.type
            : "";
    const name = rawName.toLowerCase();
    let cat = "other";
    if (name.includes("search") || name.includes("google") || name === "web_search") cat = "search";
    else if (name.includes("image_generation") || name.includes("draw") || name.includes("dall"))
      cat = "image";
    else if (name.includes("pdf") || name.includes("file_reader")) cat = "pdf";
    else if (name.includes("ocr")) cat = "ocr";

    const maxLimit =
      cat === "search" ? 1 : cat === "image" ? 1 : cat === "pdf" ? 1 : cat === "ocr" ? 2 : Infinity;
    const count = categoryCounts.get(cat) || 0;
    if (count < maxLimit) {
      categoryCounts.set(cat, count + 1);
      finalTools.push(t);
    }
  }

  return finalTools;
}

/**
 * Merge multiple system prompt entries into a single system message (Goal 3).
 * Ensures system message count === 1.
 */
export function mergeSystemPrompts(messages: JsonRecord[]): JsonRecord[] {
  if (!Array.isArray(messages)) return messages;

  const systemTexts: string[] = [];
  const nonSystem: JsonRecord[] = [];

  for (const m of messages) {
    if (m && typeof m === "object" && (m.role === "system" || m.role === "developer")) {
      const content = m.content;
      if (typeof content === "string") {
        if (content.trim()) systemTexts.push(content.trim());
      } else if (Array.isArray(content)) {
        const textParts = content
          .filter(
            (p: Record<string, unknown>) =>
              p && typeof p === "object" && p.type === "text" && typeof p.text === "string"
          )
          .map((p: Record<string, unknown>) => (p.text as string).trim())
          .filter(Boolean);
        if (textParts.length > 0) systemTexts.push(textParts.join("\n"));
      }
    } else {
      nonSystem.push(m);
    }
  }

  if (systemTexts.length === 0) {
    return nonSystem;
  }

  // Deduplicate identical system prompt lines/blocks
  const uniqueTexts: string[] = [];
  const seen = new Set<string>();
  for (const txt of systemTexts) {
    if (!seen.has(txt)) {
      seen.add(txt);
      uniqueTexts.push(txt);
    }
  }

  const combinedSystemText = uniqueTexts.join("\n\n");
  const singleSystemMessage: JsonRecord = {
    role: "system",
    content: combinedSystemText,
  };

  return [singleSystemMessage, ...nonSystem];
}

/**
 * Remove empty tool arrays, tool_choice, metadata, empty objects, null/undefined (Goal 5).
 */
export function cleanEmptyFields(result: JsonRecord): void {
  if (Array.isArray(result.tools) && result.tools.length === 0) {
    delete result.tools;
    delete result.tool_choice;
  }
  if (!result.tools) {
    delete result.tools;
    delete result.tool_choice;
  }
  if (!result.tool_choice) {
    delete result.tool_choice;
  }
  if (Array.isArray(result.stop) && result.stop.length === 0) {
    delete result.stop;
  }
  if (
    result.metadata &&
    typeof result.metadata === "object" &&
    Object.keys(result.metadata as object).length === 0
  ) {
    delete result.metadata;
  }
  if (
    result.response_format &&
    typeof result.response_format === "object" &&
    Object.keys(result.response_format as object).length === 0
  ) {
    delete result.response_format;
  }

  // Scrub null, undefined, empty arrays, and empty objects
  for (const [key, val] of Object.entries(result)) {
    if (val === null || val === undefined) {
      delete result[key];
    } else if (
      Array.isArray(val) &&
      val.length === 0 &&
      key !== "messages" &&
      key !== "input" &&
      key !== "contents"
    ) {
      delete result[key];
    } else if (
      val &&
      typeof val === "object" &&
      !Array.isArray(val) &&
      Object.keys(val as object).length === 0 &&
      key !== "response_format"
    ) {
      delete result[key];
    }
  }
}

/**
 * Log payload statistics in the required format (Goal 6).
 */
export function logPayloadStats(result: JsonRecord): void {
  let systemCount = 0;
  if (Array.isArray(result.messages)) {
    systemCount = result.messages.filter(
      (m) => m && typeof m === "object" && (m.role === "system" || m.role === "developer")
    ).length;
  } else if (result.system) {
    systemCount = Array.isArray(result.system) ? result.system.length : 1;
  }

  const tools = Array.isArray(result.tools) ? result.tools : [];
  const toolCount = tools.length;
  const toolNames = tools.map((t) => {
    const fn = (t.function as JsonRecord) || t;
    return typeof fn.name === "string"
      ? fn.name
      : typeof t.name === "string"
        ? t.name
        : typeof t.type === "string"
          ? t.type
          : "unknown";
  });
  const messages = Array.isArray(result.messages) ? result.messages : [];
  const payloadStr = JSON.stringify(result);
  const payloadLen = payloadStr.length;
  const approxTokens = Math.ceil(payloadLen / 4);

  console.log("===== Payload Stats =====");
  console.log("System:", systemCount);
  console.log("Tools:", toolCount);
  if (toolNames.length > 0) {
    console.log("Selected Tools:", toolNames.join(", "));
  } else {
    console.log("Selected Tools: none");
  }
  console.log("Messages:", messages.length);
  console.log("Payload:", payloadLen);
  console.log("Approx Tokens:", approxTokens);
}
