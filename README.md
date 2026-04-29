# memory-tdai-patch-for-MultiAgent

Patch for memory-tdai L1→L2 sender identity fix, designed for multi-Agent environments.

## Problem

In a multi-Agent system (e.g., 封神榜 / BB project), the L1→L2 memory pipeline conflated sender identities — real users (Merlin) and AI Agents (BB, Meltryllis, etc.) were all stored under the same `user` label in L2, making it impossible to distinguish who said/did what.

## Solution

Three files patched to carry sender identity through the full L0→L1→L2 pipeline:

| File | Change |
|------|--------|
| `src/conversation/l0-recorder.ts` | `sender_label` field added to `ConversationMessage` and `L0MessageRecord` |
| `src/scene/scene-extractor.ts` | `sender_label` and `role` fields added to memory records |
| `src/prompts/scene-extraction.ts` | Sender identity recognition rules inserted into scene extraction prompt |

## Usage

Copy these three files into your `memory-tdai` plugin source directory:

```bash
# Backup originals first
cp src/conversation/l0-recorder.ts src/conversation/l0-recorder.ts.bak
cp src/scene/scene-extractor.ts src/scene/scene-extractor.ts.bak
cp src/prompts/scene-extraction.ts src/prompts/scene-extraction.ts.bak

# Apply patch
cp l0-recorder.ts <your-memory-tdai>/src/conversation/
cp scene-extractor.ts <your-memory-tdai>/src/scene/
cp scene-extraction.ts <your-memory-tdai>/src/prompts/

# Restart OpenClaw gateway
openclaw gateway restart
```

## Compatibility

- **memory-tdai** plugin v3.x
- **OpenClaw** 2026.4+
- Works with Qdrant + Redis backend

## Credits

- BB (Caster class AI Agent) — patch author
- Merlin (@merlin:matrix.zerotier.local) — project owner

## License

MIT
