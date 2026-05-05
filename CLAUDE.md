# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a spec-driven development workspace using the `openspec` CLI. It manages changes through a structured proposal → implementation → archive lifecycle.

## OpenSpec CLI Commands

```bash
openspec list --json                              # list active changes
openspec new change "<name>"                      # create a new change (kebab-case name)
openspec status --change "<name>" --json          # check artifact completion and schema info
openspec instructions <artifact-id> --change "<name>" --json  # get creation instructions for an artifact
openspec instructions apply --change "<name>" --json          # get implementation instructions + task list
```

## Workflow

Changes follow the `spec-driven` schema defined in `openspec/config.yaml`.

**Artifact order** (each depends on the previous):
1. `proposal.md` — what & why
2. `design.md` — how
3. `tasks.md` — implementation checklist (`- [ ]` / `- [x]`)

**Lifecycle:**
- Active changes: `openspec/changes/<name>/`
- Archived: `openspec/changes/archive/YYYY-MM-DD-<name>/`

## Slash Commands

| Command | Purpose |
|---|---|
| `/opsx:propose` | Create change + generate all artifacts in one step |
| `/opsx:apply` | Implement tasks from a change |
| `/opsx:explore` | Thinking/exploration mode (read-only — no code implementation) |
| `/opsx:archive` | Move completed change to archive |

## Key Conventions

- `openspec instructions` returns `context` and `rules` as constraints for the AI — never copy them into artifact files.
- Task completion: toggle `- [ ]` → `- [x]` immediately after finishing each task.
- When `openspec instructions apply` returns `state: "blocked"`, artifacts are missing — use `/opsx:propose` to create them first.
- When `state: "all_done"`, suggest `/opsx:archive`.
- Delta specs (per-change) live at `openspec/changes/<name>/specs/`; main specs live at `openspec/specs/`. Archive step handles syncing.
