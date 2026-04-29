/**
 * SceneExtractor: LLM-driven memory extraction into scene blocks.
 *
 * Replaces the keyword-based SceneManager.processNewMemories() with an
 * LLM agent that autonomously reads/writes scene block files using tools.
 *
 * Security: The LLM is sandboxed — workspaceDir is set to scene_blocks/
 * so it can ONLY operate on .md scene files. System files (checkpoint,
 * scene_index, persona.md) are physically invisible to the LLM.
 *
 * Flow:
 *   1. Backup + load scene index + build summaries
 *   2. Assemble extraction prompt with memories + scene context
 *   3. Run via CleanContextRunner (tools enabled, sandboxed to scene_blocks/)
 *   4. Cleanup: remove soft-deletes, sync index, update navigation
 *   5. Parse LLM text output for out-of-band persona update signals
 */

import fs from "node:fs/promises";
import path from "node:path";
import { CleanContextRunner } from "../utils/clean-context-runner.js";
import { CheckpointManager } from "../utils/checkpoint.js";
import { BackupManager } from "../utils/backup.js";
import { readSceneIndex, syncSceneIndex } from "../scene/scene-index.js";
import type { SceneIndexEntry } from "../scene/scene-index.js";
import { generateSceneNavigation, stripSceneNavigation } from "../scene/scene-navigation.js";
import { buildSceneExtractionPrompt } from "../prompts/scene-extraction.js";

const TAG = "[memory-tdai] [extractor]";

interface ExtractorLogger {
  debug?: (message: string) => void;
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
}

export interface ExtractionResult {
  memoriesProcessed: number;
  success: boolean;
  error?: string;
}

export interface SceneExtractorOptions {
  dataDir: string;
  config: unknown;
  model?: string;
  maxScenes?: number;
  sceneBackupCount?: number;
  timeoutMs?: number;
  logger?: ExtractorLogger;
}

/**
 * Parse LLM text output for a persona update request signal.
 *
 * Supports multiple formats for robustness:
 * - Block: [PERSONA_UPDATE_REQUEST]reason: xxx[/PERSONA_UPDATE_REQUEST]
 * - Inline: PERSONA_UPDATE_REQUEST: xxx
 */
export function parsePersonaUpdateSignal(text: string): { reason: string } | null {
  // Block format: [PERSONA_UPDATE_REQUEST]...[/PERSONA_UPDATE_REQUEST]
  const blockMatch = text.match(
    /\[PERSONA_UPDATE_REQUEST\]\s*(?:reason:\s*)?(.+?)\s*\[\/PERSONA_UPDATE_REQUEST\]/s,
  );
  if (blockMatch) return { reason: blockMatch[1]!.trim() };

  // Inline format: PERSONA_UPDATE_REQUEST: reason text
  const inlineMatch = text.match(
    /PERSONA_UPDATE_REQUEST:\s*(.+?)(?:\n|$)/,
  );
  if (inlineMatch) return { reason: inlineMatch[1]!.trim() };

  return null;
}

export class SceneExtractor {
  private dataDir: string;
  private runner: CleanContextRunner;
  private maxScenes: number;
  private sceneBackupCount: number;
  private timeoutMs: number;
  private logger: ExtractorLogger | undefined;

  constructor(opts: SceneExtractorOptions) {
    this.dataDir = opts.dataDir;
    this.maxScenes = opts.maxScenes ?? 20;
    this.sceneBackupCount = opts.sceneBackupCount ?? 10;
    this.timeoutMs = opts.timeoutMs ?? 300_000; // 5 min — LLM may do multiple tool calls
    this.logger = opts.logger;

    this.runner = new CleanContextRunner({
      config: opts.config,
      modelRef: opts.model,
      enableTools: true,
      logger: opts.logger,
    });

    this.logger?.debug?.(`${TAG} Created: dataDir=${opts.dataDir}, model=${opts.model ?? "(default)"}, maxScenes=${this.maxScenes}, timeout=${this.timeoutMs}ms`);
  }

  /**
   * Extract a batch of memories into scene blocks using the LLM agent.
   *
   * @param memories - Array of raw memory records from the API
   * @returns Extraction result with count and success flag
   */
  async extract(memories: Array<{ content: string; created_at: string; id?: string }>): Promise<ExtractionResult> {
    const extractStartMs = Date.now();
    this.logger?.info(`${TAG} extract() start: ${memories.length} memories`);

    if (memories.length === 0) {
      this.logger?.debug?.(`${TAG} extract() skipped: no memories`);
      return { memoriesProcessed: 0, success: true };
    }

    const sceneBlocksDir = path.join(this.dataDir, "scene_blocks");
    const metadataDir = path.join(this.dataDir, ".metadata");

    // Ensure directories exist
    await fs.mkdir(sceneBlocksDir, { recursive: true });
    await fs.mkdir(metadataDir, { recursive: true });

    // Phase 1: Backup
    const backupStartMs = Date.now();
    const cpManager = new CheckpointManager(this.dataDir);
    const cp = await cpManager.read();
    const bm = new BackupManager(path.join(this.dataDir, ".backup"));
    await bm.backupDirectory(sceneBlocksDir, "scene_blocks", `offset${cp.total_processed}`, this.sceneBackupCount);
    this.logger?.debug?.(`${TAG} extract() backup phase: ${Date.now() - backupStartMs}ms`);

    // Phase 2: Load scene index
    const indexStartMs = Date.now();
    const index = await readSceneIndex(this.dataDir);
    this.logger?.debug?.(`${TAG} extract() scene index loaded: ${index.length} entries (${Date.now() - indexStartMs}ms)`);

    // Build scene summaries for the prompt (relative filenames only)
    const { summaries: sceneSummaries, filenames: existingSceneFiles } =
      this.buildSceneSummaries(index);

    // Build scene count warning (tiered system)
    let sceneCountWarning: string | undefined;
    const sceneCount = index.length;
    if (sceneCount >= this.maxScenes) {
      sceneCountWarning = `当前场景数量为 **${sceneCount} 个**，已达到或超过 ${this.maxScenes} 个上限！\n**你必须先执行 MERGE 操作**，将最相似的 2-4 个场景合并为 1 个，然后再处理新记忆。\n参考合并对象：热度最低或主题高度重叠的场景。`;
      this.logger?.warn(`${TAG} extract() scene count at limit: ${sceneCount}/${this.maxScenes}`);
    } else if (sceneCount === this.maxScenes - 1) {
      sceneCountWarning = `当前场景数量为 **${sceneCount} 个**，距离上限只差 1 个！\n本次处理**只能 UPDATE 现有场景，不能 CREATE 新场景**。`;
      this.logger?.warn(`${TAG} extract() scene count near limit (CREATE blocked): ${sceneCount}/${this.maxScenes}`);
    } else if (sceneCount >= this.maxScenes - 3) {
      sceneCountWarning = `当前场景数量为 **${sceneCount} 个**，建议优先考虑 UPDATE 或主动 MERGE 相似场景。`;
      this.logger?.debug?.(`${TAG} extract() scene count approaching limit: ${sceneCount}/${this.maxScenes}`);
    }

    // Phase 3: Build prompt
    const promptStartMs = Date.now();
    const memoriesJson = JSON.stringify(
      memories.map((m) => ({
        content: m.content,
        created_at: m.created_at,
        id: m.id ?? "",
        sender_label: (m as Record<string, unknown>).sender_label as string ?? "",
        role: (m as Record<string, unknown>).role as string ?? "",
      })),
      null,
      2,
    );

    const currentTimestamp = formatTimestamp(new Date());

    const prompt = buildSceneExtractionPrompt({
      memoriesJson,
      sceneSummaries: sceneSummaries || "(无已有场景)",
      currentTimestamp,
      sceneCountWarning,
      existingSceneFiles,
    });
    this.logger?.debug?.(`${TAG} extract() prompt built: ${prompt.length} chars (${Date.now() - promptStartMs}ms)`);

    // Phase 4: Run LLM agent (sandboxed to scene_blocks/)
    let llmOutput = "";
    try {
      this.logger?.debug?.(`${TAG} extract() starting LLM runner (timeout=${this.timeoutMs}ms, maxTokens=model default)...`);
      const runnerStartMs = Date.now();
      llmOutput = await this.runner.run({
        prompt,
        taskId: `scene-extract-${Date.now()}`,
        timeoutMs: this.timeoutMs,
        // maxTokens omitted → core uses the resolved model's maxTokens from catalog
        workspaceDir: sceneBlocksDir,
      }) ?? "";
      this.logger?.debug?.(`${TAG} extract() LLM runner completed: ${Date.now() - runnerStartMs}ms`);
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      const totalMs = Date.now() - extractStartMs;
      this.logger?.error(`${TAG} extract() LLM runner failed after ${totalMs}ms: ${errMsg}`);
      return { memoriesProcessed: 0, success: false, error: errMsg };
    }

    // Phase 5: Subsequent processing — safe cleanup of soft-deleted files
    //
    // Security: The LLM has no `exec` tool and cannot run shell commands.
    // Instead, it "deletes" files by writing empty content (soft-delete).
    // Here we detect and remove those empty files before syncing the index,
    // so syncSceneIndex won't re-index stale empty entries.
    const cleanupStartMs = Date.now();
    let cleanedCount = 0;
    try {
      const allFiles = (await fs.readdir(sceneBlocksDir)).filter((f) => f.endsWith(".md"));
      for (const file of allFiles) {
        const filePath = path.join(sceneBlocksDir, file);
        const content = await fs.readFile(filePath, "utf-8");
        if (content.trim().length === 0) {
          await fs.unlink(filePath);
          cleanedCount++;
          this.logger?.debug?.(`${TAG} extract() removed soft-deleted file: ${file}`);
        }
      }
    } catch (cleanupErr) {
      // Non-fatal — log and continue to index sync
      this.logger?.warn(`${TAG} extract() soft-delete cleanup error: ${cleanupErr instanceof Error ? cleanupErr.message : String(cleanupErr)}`);
    }
    this.logger?.debug?.(`${TAG} extract() soft-delete cleanup: removed ${cleanedCount} empty files (${Date.now() - cleanupStartMs}ms)`);

    // Phase 6: Sync scene index (rebuilds from remaining non-empty files)
    const syncStartMs = Date.now();
    await syncSceneIndex(this.dataDir);
    this.logger?.debug?.(`${TAG} extract() scene index synced: ${Date.now() - syncStartMs}ms`);

    // Phase 7: Update persona.md navigation (GAP-4 fix)
    const navStartMs = Date.now();
    try {
      await this.updateSceneNavigation();
      this.logger?.debug?.(`${TAG} extract() persona.md navigation updated: ${Date.now() - navStartMs}ms`);
    } catch (navErr) {
      // Non-fatal — log and continue
      this.logger?.warn(`${TAG} extract() failed to update persona navigation: ${navErr instanceof Error ? navErr.message : String(navErr)}`);
    }

    // Phase 8: Parse LLM output for out-of-band persona update signal
    if (llmOutput) {
      const signal = parsePersonaUpdateSignal(llmOutput);
      if (signal) {
        await cpManager.setPersonaUpdateRequest(signal.reason);
        this.logger?.debug?.(`${TAG} extract() persona update requested by LLM: ${signal.reason}`);
      }
    }

    const totalMs = Date.now() - extractStartMs;
    this.logger?.info(`${TAG} extract() completed: ${memories.length} memories processed in ${totalMs}ms`);

    return { memoriesProcessed: memories.length, success: true };
  }

  /**
   * Build human-readable scene summaries for the prompt,
   * and collect the list of existing scene filenames (relative).
   *
   * Includes a capacity counter at the top (e.g. "当前场景总数：5 / 15")
   * so the LLM can immediately see how close it is to the limit.
   */
  private buildSceneSummaries(
    index: SceneIndexEntry[],
  ): { summaries: string; filenames: string[] } {
    if (index.length === 0) return { summaries: "", filenames: [] };

    const lines: string[] = [];
    const filenames: string[] = [];

    // Inject capacity counter at the top — LLM sees this first
    lines.push(`**当前场景总数：${index.length} / ${this.maxScenes}**`);
    lines.push("");

    for (const entry of index) {
      filenames.push(entry.filename);
      lines.push(`### ${entry.filename}`);
      lines.push(`**热度**: ${entry.heat} | **更新**: ${entry.updated}`);
      lines.push(`**summary**: ${entry.summary}`);
      lines.push("");
    }
    return { summaries: lines.join("\n"), filenames };
  }

  /**
   * Update the scene navigation section at the end of persona.md.
   *
   * Reads the current scene index, generates the navigation block, then
   * strips any existing navigation from persona.md and appends the new one.
   *
   * IMPORTANT: If the persona body is empty (PersonaGenerator hasn't run yet),
   * we skip writing to avoid creating a persona.md that only contains the
   * scene navigation. PersonaGenerator.generate() will write the full
   * persona + navigation when it runs.
   */
  private async updateSceneNavigation(): Promise<void> {
    const personaPath = path.join(this.dataDir, "persona.md");
    const index = await readSceneIndex(this.dataDir);
    const nav = generateSceneNavigation(index);

    let existing = "";
    try {
      existing = await fs.readFile(personaPath, "utf-8");
    } catch {
      // No persona file yet — PersonaGenerator will create it with navigation.
      // Don't write a navigation-only file.
      this.logger?.debug?.(`${TAG} updateSceneNavigation() skipped: no persona file yet, waiting for PersonaGenerator`);
      return;
    }

    if (!existing.trim() && !nav) return;

    const stripped = stripSceneNavigation(existing).trimEnd();

    // If the persona body is empty (only navigation existed), don't overwrite
    // with a navigation-only file. Let PersonaGenerator handle full generation.
    if (!stripped) {
      this.logger?.debug?.(`${TAG} updateSceneNavigation() skipped: persona body is empty, waiting for PersonaGenerator`);
      return;
    }

    const updated = nav ? `${stripped}\n\n${nav}\n` : `${stripped}\n`;

    // persona.md is at dataDir root, no subdir needed
    await fs.writeFile(personaPath, updated, "utf-8");
  }
}

function formatTimestamp(d: Date): string {
  return d.toISOString();
}
