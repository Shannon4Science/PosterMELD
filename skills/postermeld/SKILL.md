---
name: postermeld
description: Install and run the complete PosterMELD multi-agent paper-to-poster pipeline to generate polished, editable PPTX posters and matching PNG previews. Use when a user asks to turn a research PDF into an academic poster, generate poster variants, apply PosterMELD templates and styles, include institution or venue logos, create a teaser or generated background, or inspect PosterMELD output. Preserve the full runtime and quality-review loop; never replace it with a simplified renderer.
---

# PosterMELD

Use this skill as a thin interface to the complete PosterMELD runtime. Do not recreate poster planning, templates, rendering, logo handling, teaser generation, backgrounds, or quality review inside the skill.

## Resolve the launcher

Resolve `SKILL_DIR` from the directory containing this `SKILL.md`. Common locations are
`$CODEX_HOME/skills/postermeld`, `~/.codex/skills/postermeld`,
`~/.claude/skills/postermeld`, and `<repository>/skills/postermeld`. Then set:

```bash
LAUNCHER="$SKILL_DIR/scripts/postermeld_skill.py"
```

## Workflow

1. Run the strict preflight before generation:

   ```bash
   python "$LAUNCHER" doctor --strict
   ```

2. If the runtime is absent, install it once:

   ```bash
   python "$LAUNCHER" install
   ```

3. If preflight reports missing configuration, read [configuration.md](references/configuration.md). Do not silently disable requested teaser, background, Logo, MinerU, or VLM stages. Full visual quality takes precedence over producing a degraded artifact.

4. Determine the paper PDF, venue, desired orientation, template/style preferences, and any user-provided Logo paths. Do not guess a venue when it is not supported by the paper path or user request.

5. Generate with the complete pipeline. Arguments after `--` are passed directly to PosterMELD:

   ```bash
   python "$LAUNCHER" run \
     --paper "/absolute/path/to/paper.pdf" \
     --output-root "/absolute/path/to/output" \
     --strict \
     -- \
     --conference AAAI \
     --poster-style navy_serif \
     --background-style auto \
     --background-palette auto
   ```

6. Keep the launcher attached until completion. Report the final `.png`, editable `.pptx`, `skill_run.log`, and quality reports. Inspect the final PNG visually when the host supports image viewing.

7. If the pipeline fails a quality gate, report the exact failed stage. Do not bypass the gate or construct a replacement poster manually.

## Default quality profile

The launcher enables the original high-quality path by default:

- automatic template selection and rich visual density;
- institution Logo resolution;
- paper-conditioned teaser and background generation;
- block-level VLM review;
- visual-legibility and global layout review;
- editable PPTX and matching PNG rendering.

User arguments supplied after `--` may override the default template, style, density, conference, Logo, header, and background controls. See [options.md](references/options.md) for common controls.

## Hard rules

- Always call `poster_generation/src/workflow/pipeline.py` through the launcher.
- Never use a skill-local renderer, fake template, flattened PPTX, or procedural substitute.
- Never print, copy into reports, or commit API credentials.
- Never treat placeholder images or silent image-generation fallback as successful full-quality output.
- Keep the original runtime code unchanged unless the user explicitly asks for a pipeline change.
