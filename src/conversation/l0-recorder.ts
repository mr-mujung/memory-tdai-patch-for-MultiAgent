/**
 * L0 Conversation Recorder: records raw conversation messages to local JSONL files.
 *
 * Triggered from agent_end hook. Receives the conversation messages directly from
 * the hook context (no file I/O needed), sanitizes them, filters out noise, and
 * writes to ~/.openclaw/memory-tdai/conversations/YYYY-MM-DD.jsonl
 *
 * Design decisions:
 * - Uses JSONL format (**one message per line** — flat, easy to grep/stream)
 * - One file per day (all sessions merged into the same daily file)
 * - sessionKey is stored as a field in each JSONL line, not in the filename
 * - Independent from system session files — format fully controlled by plugin
 * - Messages are sanitized to remove injected tags (prevent feedback loops)
 * - Short/long/command messages are filtered out
 */

import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { sanitizeText, shouldCaptureL0 } from "../utils/sanitize.js";

// ============================
// Types
// ============================

export interface ConversationMessage {
  /** Unique message ID (used by L1 prompt for source_message_ids tracking) */
  id: string;
  role: "user" | "assistant";
  /** Sender identity — username for real users, agent name for AIs */
  sender_label?: string;
  content: string;
  timestamp: number; // epoch ms
}

/**
 * Generate a short unique message ID.
 */
function generateMessageId(): string {
  return `msg_${Date.now()}_${crypto.randomBytes(3).toString("hex")}`;
}

/**
 * New flat format: one message per JSONL line.
 */
export interface L0MessageRecord {
  sessionKey: string;
  sessionId: string;
  recordedAt: string; // ISO timestamp
  id: string;
  role: "user" | "assistant";
  sender_label?: string;
  content: string;
  timestamp: number; // epoch ms
}

/**
 * A group of conversation messages (used by downstream consumers).
 * Each L0ConversationRecord represents one or more messages from the same recording event.
 */
export interface L0ConversationRecord {
  sessionKey: string;
  sessionId: string;
  recordedAt: string; // ISO timestamp
  messageCount: number;
  messages: ConversationMessage[];
}

interface Logger {
  debug?: (message: string) => void;
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
}

const TAG = "[memory-tdai][l0]";

// ============================
// Core function
// ============================

/**
 * Record a conversation round to the L0 JSONL file.
 *
 * Only records **incremental** messages (new since the last capture).
 * Uses `afterTimestamp` as the primary filter to skip already-captured history.
 *
 * @param sessionKey - The session key for this conversation
 * @param rawMessages - Raw messages from the agent_end hook context (full session history)
 * @param baseDir - Base data directory (~/.openclaw/memory-tdai/)
 * @param logger - Optional logger
 * @param originalUserText - Clean original user prompt (pre-prependContext)
 * @param afterTimestamp - Epoch ms cursor: only messages with timestamp > this are new.
 *                         Pass 0 or omit for the first capture of a session.
 * @returns Filtered messages (for L1 to use directly), or empty array if nothing worth recording
 */
export async function recordConversation(params: {
  sessionKey: string;
  sessionId?: string;
  rawMessages: unknown[];
  baseDir: string;
  logger?: Logger;
  /** Clean original user prompt (pre-prependContext) */
  originalUserText?: string;
  /** Epoch ms cursor: only process messages with timestamp strictly greater than this. */
  afterTimestamp?: number;
  /**
   * Number of messages in the session at before_prompt_build time.
   * Used to locate the exact user message that originalUserText corresponds to:
   * rawMessages[originalUserMessageCount] is the user message appended by the framework
   * AFTER before_prompt_build, i.e. the one whose content was polluted by prependContext.
   */
  originalUserMessageCount?: number;
}): Promise<ConversationMessage[]> {
  const { sessionKey, sessionId, rawMessages, baseDir, logger, originalUserText, afterTimestamp, originalUserMessageCount } = params;

  // Step 1: Extract user/assistant messages from raw hook messages
  const allExtracted = extractUserAssistantMessages(rawMessages);
  logger?.debug?.(`${TAG} Extracted ${allExtracted.length} user/assistant messages from ${rawMessages.length} total`);

  // Step 1.5: Incremental filter — only keep messages newer than the cursor.
  //
  // Uses strict greater-than (>) which is safe because:
  //   - The cursor is set to max(timestamps) of the LAST recorded batch.
  //   - The next agent turn's messages will have timestamps strictly greater than
  //     the previous turn (there's at least one LLM API call between turns, which
  //     takes hundreds of milliseconds minimum — no same-millisecond collision).
  //   - All messages within a single turn are captured together as one batch,
  //     so even if multiple messages share the same timestamp, they are either
  //     all included (new batch) or all excluded (already captured).
  //   - If a message lacks a timestamp field, extractUserAssistantMessages()
  //     assigns Date.now() at extraction time, which is always > previous cursor.
  const cursor = afterTimestamp ?? 0;
  const extracted = cursor > 0
    ? allExtracted.filter((m) => m.timestamp > cursor)
    : allExtracted;

  if (cursor > 0) {
    logger?.debug?.(
      `${TAG} Incremental filter: ${allExtracted.length} total → ${extracted.length} new (cursor=${cursor})`,
    );
  }

  if (extracted.length === 0) {
    logger?.debug?.(`${TAG} No new user/assistant messages to record`);
    return [];
  }

  // Step 2: Replace polluted user messages with cached original prompt.
  //
  // Background:
  //   The framework appends the user's message to the session after before_prompt_build,
  //   then injects prependContext into it. So the user message in rawMessages is polluted.
  //   We cached the clean prompt (originalUserText) and the message count at
  //   before_prompt_build time (originalUserMessageCount) to identify which raw message
  //   is the real user input.
  //
  // Strategy:
  //   Use rawMessages[originalUserMessageCount] to find the timestamp of the polluted
  //   user message, then match it in `extracted` by timestamp for replacement.
  //   This is precise — no ambiguity even with tool_result user messages.
  //   If matching fails (missing messageCount, no timestamp, or filtered by cursor),
  //   skip replacement entirely — sanitizeText() in Step 3 is the safety net.
  if (originalUserText && originalUserMessageCount != null && originalUserMessageCount >= 0 && originalUserMessageCount < rawMessages.length) {
    const targetRaw = rawMessages[originalUserMessageCount] as Record<string, unknown> | undefined;
    const targetTs = targetRaw && typeof targetRaw.timestamp === "number" ? targetRaw.timestamp : undefined;

    if (targetTs != null) {
      let replaced = false;
      for (let i = 0; i < extracted.length; i++) {
        if (extracted[i].role === "user" && extracted[i].timestamp === targetTs) {
          logger?.debug?.(
            `${TAG} Replacing user message at timestamp=${targetTs} with cached original prompt ` +
            `(${originalUserText.length} chars, was ${extracted[i].content.length} chars) [matched by messageCount=${originalUserMessageCount}]`,
          );
          extracted[i] = { ...extracted[i], content: originalUserText };
          replaced = true;
          break;
        }
      }
      if (!replaced) {
        logger?.warn?.(
          `${TAG} Target user message (ts=${targetTs}, rawIdx=${originalUserMessageCount}) not found in extracted batch — ` +
          `possibly filtered by cursor. Skipping replacement, will rely on sanitizeText().`,
        );
      }
    } else {
      logger?.warn?.(
        `${TAG} rawMessages[${originalUserMessageCount}] has no valid timestamp — ` +
        `skipping replacement, will rely on sanitizeText().`,
      );
    }
  } else if (originalUserText) {
    logger?.warn?.(
      `${TAG} Have originalUserText but no valid originalUserMessageCount — ` +
      `skipping replacement, will rely on sanitizeText().`,
    );
  }

  // Step 3: Sanitize and filter
  const filtered = extracted
    .map((m) => ({
      id: m.id,
      role: m.role,
      content: sanitizeText(m.content),
      timestamp: m.timestamp,
    }))
    .filter((m) => shouldCaptureL0(m.content));

  logger?.debug?.(`${TAG} After sanitize+filter: ${filtered.length} messages (from ${extracted.length})`);

  if (filtered.length === 0) {
    logger?.debug?.(`${TAG} All messages filtered out, skipping L0 write`);
    return [];
  }

  // Step 4: Write to JSONL file — one message per line (flat format)
  const now = new Date().toISOString();
  const lines: string[] = [];
  for (const msg of filtered) {
    const record: L0MessageRecord = {
      sessionKey,
      sessionId: sessionId || "",
      recordedAt: now,
      id: msg.id,
      role: msg.role,
      sender_label: msg.sender_label,
      content: msg.content,
      timestamp: msg.timestamp,
    };
    lines.push(JSON.stringify(record));
  }

  const shardDate = formatLocalDate(new Date());
  const outDir = path.join(baseDir, "conversations");
  const outPath = path.join(outDir, `${shardDate}.jsonl`);

  try {
    await fs.mkdir(outDir, { recursive: true });
    // Append each message as its own JSONL line
    await fs.appendFile(outPath, lines.join("\n") + "\n", "utf-8");
    logger?.debug?.(`${TAG} Recorded ${filtered.length} messages to ${outPath}`);
  } catch (err) {
    logger?.error(`${TAG} Failed to write L0 file: ${err instanceof Error ? err.message : String(err)}`);
    // Return filtered messages anyway so L1 can still process them
  }

  return filtered;
}

/**
 * Read all L0 conversation records for a session.
 * Returns records in chronological order.
 *
 * File format: `YYYY-MM-DD.jsonl` (daily files, all sessions merged).
 * Each line is an L0MessageRecord; filtered by sessionKey at line level.
 */
export async function readConversationRecords(
  sessionKey: string,
  baseDir: string,
  logger?: Logger,
): Promise<L0ConversationRecord[]> {
  const conversationsDir = path.join(baseDir, "conversations");

  // Daily file pattern: YYYY-MM-DD.jsonl
  const dateFilePattern = /^\d{4}-\d{2}-\d{2}\.jsonl$/;

  let entries: string[];
  try {
    const dirEntries = await fs.readdir(conversationsDir, { withFileTypes: true });
    entries = dirEntries
      .filter((entry) => entry.isFile())
      .map((entry) => entry.name);
  } catch {
    // Directory doesn't exist yet — normal for first conversation
    return [];
  }

  const targetFiles = entries
    .filter((name) => dateFilePattern.test(name))
    .sort();

  if (targetFiles.length === 0) {
    return [];
  }

  const records: L0ConversationRecord[] = [];

  for (const fileName of targetFiles) {
    const filePath = path.join(conversationsDir, fileName);

    let raw: string;
    try {
      raw = await fs.readFile(filePath, "utf-8");
    } catch {
      logger?.warn?.(`${TAG} Failed to read L0 file: ${filePath}`);
      continue;
    }

    const lines = raw.split("\n").filter((line: string) => line.trim());
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      try {
        const parsed = JSON.parse(line) as Record<string, unknown>;

        // Filter by sessionKey at line level
        const lineSessionKey = parsed.sessionKey as string | undefined;
        if (lineSessionKey !== sessionKey) continue;

        if (typeof parsed.role === "string" && typeof parsed.content === "string") {
          // Flat format: { sessionKey, sessionId, recordedAt, id, role, content, timestamp }
          // Wrap into L0ConversationRecord for uniform downstream consumption
          const msg: ConversationMessage = {
            id: (typeof parsed.id === "string" && parsed.id) ? parsed.id : generateMessageId(),
            role: parsed.role as "user" | "assistant",
            sender_label: typeof parsed.sender_label === "string" ? parsed.sender_label : undefined,
            content: parsed.content as string,
            timestamp: typeof parsed.timestamp === "number" ? parsed.timestamp : Date.now(),
          };
          records.push({
            sessionKey: (parsed.sessionKey as string) || sessionKey,
            sessionId: (parsed.sessionId as string) || "",
            recordedAt: (parsed.recordedAt as string) || new Date().toISOString(),
            messageCount: 1,
            messages: [msg],
          });
        } else {
          logger?.warn?.(`${TAG} Unrecognized JSONL line format in ${filePath}:${i + 1}`);
        }
      } catch {
        logger?.warn?.(`${TAG} Skipping malformed JSONL line in ${filePath}:${i + 1}`);
      }
    }
  }

  records.sort((a, b) => {
    const ta = Date.parse(a.recordedAt);
    const tb = Date.parse(b.recordedAt);
    const na = Number.isFinite(ta) ? ta : Number.POSITIVE_INFINITY;
    const nb = Number.isFinite(tb) ? tb : Number.POSITIVE_INFINITY;
    return na - nb;
  });

  return records;
}

/**
 * Read L0 messages across all conversation records for a session,
 * optionally filtered by a cursor timestamp (messages after the cursor).
 *
 * When `limit` is provided, only the **newest** `limit` messages are returned
 * (matching the DB path's `ORDER BY timestamp DESC LIMIT ?` behavior).
 * Returned messages are always in chronological order (oldest → newest).
 *
 * NOTE: potential optimization — records are chronologically ordered (append-only JSONL),
 * so a reverse scan could skip entire old records. Deferred for now; see Issue 5 in
 * docs/05-known-issues.md.
 */
export async function readConversationMessages(
  sessionKey: string,
  baseDir: string,
  afterTimestamp?: number,
  logger?: Logger,
  limit?: number,
): Promise<ConversationMessage[]> {
  const records = await readConversationRecords(sessionKey, baseDir, logger);
  const allMessages: ConversationMessage[] = [];

  for (const record of records) {
    for (const msg of record.messages) {
      if (afterTimestamp && msg.timestamp <= afterTimestamp) continue;
      allMessages.push(msg);
    }
  }

  // Truncate to newest `limit` messages (keep tail, since array is chronological)
  if (limit != null && limit > 0 && allMessages.length > limit) {
    logger?.debug?.(
      `${TAG} readConversationMessages: truncating ${allMessages.length} → ${limit} (newest)`,
    );
    return allMessages.slice(-limit);
  }

  return allMessages;
}

/**
 * A group of conversation messages sharing the same sessionId.
 */
export interface SessionIdMessageGroup {
  sessionId: string;
  messages: ConversationMessage[];
}

/**
 * Read L0 messages for a session, grouped by sessionId.
 *
 * Within the same sessionKey, different sessionIds represent different conversation
 * instances (e.g. after /reset). L1 extraction should process each group independently
 * so that each group's sessionId is correctly associated with its extracted memories.
 *
 * When `limit` is provided, only the **newest** `limit` messages (across all groups)
 * are retained — matching the DB path's `ORDER BY timestamp DESC LIMIT ?` behavior.
 * Groups that become empty after truncation are dropped.
 *
 * Groups are returned in chronological order (by earliest message timestamp).
 * Messages within each group are also in chronological order.
 */
export async function readConversationMessagesGroupedBySessionId(
  sessionKey: string,
  baseDir: string,
  afterTimestamp?: number,
  logger?: Logger,
  limit?: number,
): Promise<SessionIdMessageGroup[]> {
  const records = await readConversationRecords(sessionKey, baseDir, logger);

  // Collect all messages with their sessionId, respecting afterTimestamp filter
  const allMessages: Array<{ sessionId: string; msg: ConversationMessage }> = [];

  for (const record of records) {
    const sid = record.sessionId || "";
    for (const msg of record.messages) {
      if (afterTimestamp && msg.timestamp <= afterTimestamp) continue;
      allMessages.push({ sessionId: sid, msg });
    }
  }

  // Sort by timestamp ASC (chronological) — records are already roughly ordered
  // by recordedAt, but messages within may not be perfectly sorted by timestamp.
  allMessages.sort((a, b) => a.msg.timestamp - b.msg.timestamp);

  // Truncate to newest `limit` messages (keep tail)
  let selected = allMessages;
  if (limit != null && limit > 0 && allMessages.length > limit) {
    logger?.debug?.(
      `${TAG} readConversationMessagesGroupedBySessionId: truncating ${allMessages.length} → ${limit} (newest)`,
    );
    selected = allMessages.slice(-limit);
  }

  // Re-group by sessionId
  const groupMap = new Map<string, ConversationMessage[]>();
  for (const { sessionId, msg } of selected) {
    let group = groupMap.get(sessionId);
    if (!group) {
      group = [];
      groupMap.set(sessionId, group);
    }
    group.push(msg);
  }

  // Convert to array, sorted by earliest message timestamp in each group
  const groups: SessionIdMessageGroup[] = [];
  for (const [sessionId, messages] of groupMap) {
    if (messages.length > 0) {
      groups.push({ sessionId, messages });
    }
  }
  groups.sort((a, b) => a.messages[0].timestamp - b.messages[0].timestamp);

  return groups;
}

// ============================
// Helpers
// ============================

/**
 * Extract user and assistant messages from raw hook message array.
 */
function extractUserAssistantMessages(messages: unknown[]): ConversationMessage[] {
  const result: ConversationMessage[] = [];

  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    const m = msg as Record<string, unknown>;
    const role = m.role as string | undefined;

    if (role !== "user" && role !== "assistant") continue;

    let content: string | undefined;
    if (typeof m.content === "string") {
      content = m.content;
    } else if (Array.isArray(m.content)) {
      const textParts: string[] = [];
      for (const part of m.content) {
        if (
          part &&
          typeof part === "object" &&
          (part as Record<string, unknown>).type === "text"
        ) {
          const text = (part as Record<string, unknown>).text;
          if (typeof text === "string") textParts.push(text);
        }
      }
      content = textParts.join("\n");
    }

    // Strip inline base64 image data URIs that some providers embed in string content.
    // These are not useful for memory and would pollute FTS / embedding indexes.
    if (content && /data:image\/[a-z+]+;base64,/i.test(content)) {
      content = content.replace(/data:image\/[a-z+]+;base64,[A-Za-z0-9+/=]+/gi, "[image]");
    }

    if (content && content.trim()) {
      const ts = typeof m.timestamp === "number" ? m.timestamp : Date.now();
      const senderLabelRaw = m.sender_label ?? m.sender ?? m.name ?? m.author ?? "";
      const sender_label = typeof senderLabelRaw === "string" ? senderLabelRaw : "";
      result.push({
        id: (typeof m.id === "string" && m.id) ? m.id : generateMessageId(),
        role: role as "user" | "assistant",
        sender_label: sender_label || undefined,
        content: content.trim(),
        timestamp: ts,
      });
    }
  }

  return result;
}

/**
 * Format local date as YYYY-MM-DD.
 */
function formatLocalDate(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}
