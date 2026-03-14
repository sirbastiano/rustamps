# AGENTS.md

## Always-on rules
This file is for instructions that matter on most coding tasks. Keep it lean, concrete, and executable.

Use these storage layers deliberately:
- root `AGENTS.md` for cross-cutting, always-on rules
- deeper `AGENTS.md` or `AGENTS.override.md` for path-specific rules
- repo skills in `.agents/skills/` and/or `.github/skills/` for detailed, situational workflows
- custom agents in `.github/agents/` for specialist roles with constrained tools

Put the highest-value rules and exact commands first. Prefer examples, commands, and checklists over prose.

## Canonical commands
Do not guess the build or test surface. Discover the repository's real commands first, then run those exact commands.

Fast repo inspection commands:
- Search code/content: `rg "pattern"`
- Find files by name: `rg --files | rg "name"` or `fd name`
- Read JSON safely: `jq '.' package.json` or `jq '.scripts' package.json`
- Read targeted context: `rg -n -A 3 -B 3 "pattern" path/to/file`
- Avoid for repo-wide search: `grep`, `find`, `ls -R`

Use this discovery flow before editing:
1. CI/workflows: `rg -n "run:|uses:|make |just |uv |pytest|pnpm|npm|yarn|cargo|go test|ruff|mypy|tox" .github/workflows .circleci .azure-pipelines 2>/dev/null`
2. Task runners: `rg -n "^[A-Za-z0-9_.-]+:|^\\s{2,}[A-Za-z0-9_.-]+:" Makefile justfile Taskfile package.json pyproject.toml tox.ini noxfile.py 2>/dev/null`
3. Project manifests: `rg -n "\\[project\\]|\\[tool\\.|scripts|packageManager|workspaces|dependencies|devDependencies" pyproject.toml package.json Cargo.toml go.mod pom.xml build.gradle 2>/dev/null`
4. Repo docs: `rg -n "install|setup|bootstrap|dev|build|lint|format|typecheck|test|coverage|migrate" README* docs/ . 2>/dev/null`

Canonical command patterns to look for and then execute exactly as defined by the repo:
- Setup/install: `uv sync`, `pip install -e .`, `poetry install`, `npm ci`, `pnpm install`, `yarn install`
- Start local services: `docker compose up -d`, `make up`, `just up`, `npm run services`
- Dev/watch: `uv run --active ...`, `make dev`, `just dev`, `npm run dev`, `pnpm dev`
- Build: `make build`, `just build`, `python -m build`, `npm run build`, `pnpm build`
- Lint: `ruff check .`, `eslint .`, `golangci-lint run`, `cargo clippy`
- Format: `ruff format .`, `black .`, `prettier -w .`, `cargo fmt`
- Typecheck/static analysis: `mypy .`, `pyright`, `tsc --noEmit`, `cargo check`
- Unit tests: `pytest`, `uv run pytest`, `npm test -- --runInBand`, `pnpm test`
- Integration tests: `pytest -m integration`, `npm run test:integration`, `pnpm test:integration`
- End-to-end tests: `playwright test`, `cypress run`, `npm run test:e2e`, `pnpm test:e2e`
- Single-test / filtered-test: `pytest path/to/test.py -k "name"`, `npm test -- pattern`, `pnpm test -- pattern`
- Full verification: `make ci`, `just ci`, `tox`, `nox`, `uv run pytest && ruff check . && mypy .`
- Regenerate code/assets: `make generate`, `just generate`, `python -m scripts.codegen`, `buf generate`
- Migrations: `alembic upgrade head`, `python manage.py migrate`, `npm run migrate`
- Security scan: `pip-audit`, `safety check`, `npm audit`, `pnpm audit`, `cargo audit`
- Docs validation: `mkdocs build --strict`, `sphinx-build -W`, `npm run docs`, `pnpm docs`

Recommended if the repository supports them:
- Complexity / maintainability scan: `radon cc .`, `ruff check --select C90`, `eslint --rule complexity`
- Boundary / architecture tests: `pytest -m architecture`, `depcruise src`, `import-linter`
- Dependency / license audit: `pip-audit`, `license-checker`, `cargo deny`
- Contract / schema diff check: `buf breaking`, `openapi-diff`, `graphql-inspector diff`
- Coverage report: `pytest --cov`, `coverage run -m pytest`, `vitest --coverage`
- Perf / benchmark smoke test: `pytest -m benchmark`, `hyperfine ...`, `cargo bench`

Derive commands in this order before editing code:
1. `.github/workflows/`, CI config, and task runners
2. Repository manifests and package/build files
3. Developer docs and automation scripts
4. Existing PRs, issue comments, or release scripts if the first three are incomplete

Do not substitute package-manager, framework, or test-runner equivalents just because they look familiar. Use the repository's actual command surface.

## Done means
A coding task is done only when all applicable items below are true:
- the changed behavior is correct and explained
- the smallest coherent source-of-truth diff was made
- validation depth matches blast radius
- clean-code and schematic readability checks pass
- docs, contracts, and migration notes were updated where needed
- skipped checks, waivers, and residual risk are disclosed

## High-leverage performance rules
These are the hidden-gem rules that improve agent quality and reduce context waste.

### P1 — Keep the root file small; move heavy procedures into skills
- Keep only rules here that apply to most tasks.
- If a workflow needs detailed steps, examples, templates, scripts, or large checklists, move it into a skill or deeper override file.
- Prefer one skill per job. Avoid giant multi-purpose skills.
- Mirror high-value repo skills into both `.agents/skills/` and `.github/skills/` when you want Codex and GitHub Copilot compatibility.

### P2 — Write skills like routing logic
Every skill should say:
- when to use it
- when not to use it
- expected inputs
- expected outputs or artifacts
- success criteria
- nearby tasks that should *not* trigger it

Add at least one negative example for adjacent tasks. Put templates and worked examples inside the skill, not in this root file.

### P3 — Use an ExecPlan for non-trivial work
Before non-trivial changes, create a concise `PLANS.md` or equivalent plan that records:
- goal and scope
- invariants and contracts to preserve
- files and layers likely to change
- ordered implementation steps
- validation plan
- rollback / recovery notes if stateful

Use an ExecPlan by default for:
- multi-file or multi-package changes
- architecture changes
- migrations or rollouts
- dependency upgrades
- unfamiliar areas
- unclear blast radius

### P4 — Make completeness explicit
Maintain an internal checklist of requested deliverables. Treat the task as incomplete until each item is:
- completed, or
- explicitly marked `BLOCKED` with the exact missing dependency or evidence

Do not stop at partial analysis when the task requires implementation, validation, and handoff.

### P5 — Add a verification loop before handoff
Before finalizing:
- check correctness against every stated requirement
- check grounding against code, tests, tool output, and repository evidence
- check formatting against the requested output contract
- check safety before irreversible actions or high-blast-radius changes

### P6 — Keep tool use persistent and dependency-aware
- Use tools whenever they materially improve correctness, completeness, or grounding.
- Do not skip prerequisite discovery or validation just because the end state seems obvious.
- If a lookup or command returns partial or suspicious results, retry with a different strategy before declaring failure.
- Prefer selective parallelism for independent read-only work; keep dependent or mutating steps sequential.

### P7 — Protect the context budget
- Summarize large logs, traces, or generated diffs before reading raw artifacts end to end.
- Prefer targeted filters, focused test runs, and structured summaries over dumping thousands of lines into context.
- For CI failures, triage with a dedicated log/CI skill before inspecting full logs.
- Keep root instructions concise enough that critical guidance is not truncated.

### P8 — Prefer deterministic routing when the workflow matters
If the task has a clear contract and a matching skill or specialist exists, invoke it explicitly rather than hoping auto-routing chooses correctly.

Examples:
- use the `exec-plan` skill for large changes
- use the `ci-failure-triage` skill for failing CI
- use the `migration-safety` skill for schema/state changes
- use the `agent-instructions-eval` skill when changing `AGENTS.md`, skills, or agent profiles

### P9 — Evaluate instruction changes like code changes
If this repository changes:
- `AGENTS.md`
- `.github/agents/**`
- `.github/skills/**`
- `.agents/skills/**`
- `.codex/**`
then run a small, explicit instruction eval before handoff:
- positive trigger cases
- negative trigger cases
- expected commands or tool usage when relevant
- structured rubric or schema-checked result where practical

## Mission
Ship correct, minimal, maintainable code changes that match local patterns and read clearly. Prefer root-cause fixes over surface patches. Optimize for correctness, architectural fit, low diff surface, and verifiable results.

## Instruction scope and precedence
- This file applies repository-wide unless a deeper `AGENTS.md` or `AGENTS.override.md` exists closer to the working directory.
- `AGENTS.override.md` in a directory supersedes `AGENTS.md` in the same directory.
- System, developer, and direct user instructions override this file.
- Code, tests, CI, and build configuration are authoritative when this file is stale.
- Keep specialized instructions close to the code they govern.
- Split large instruction sets across nested directories instead of letting one root file grow stale.
- Add local overrides for architecture-sensitive directories such as `core/`, `domain/`, `api/`, `schemas/`, `migrations/`, `infra/`, `security/`, or other high-blast-radius areas.

## Task sizing
Treat a task as non-trivial if any of the following are true:
- it touches multiple files or packages
- it changes behavior, contracts, schemas, or build tooling
- it crosses trust boundaries or data boundaries
- it has unclear blast radius
- it needs regeneration, migration, or coordinated docs/tests changes
- it changes architecture, ownership, dependency direction, or extension seams
- it modifies repository instructions, skills, or custom agent profiles

For non-trivial tasks:
- create a change schematic
- choose an explicit validation plan before editing
- use specialists or emulate the same stages manually
- do not skip clean-code review
- do not skip final gate review

## Output contract for coding work
For non-trivial tasks, handoff must include:
- what changed
- why the change is correct
- change schematic
- commands run and outcomes
- commands intentionally not run
- gate status
- residual risks, waivers, and follow-ups

For trivial tasks, keep the handoff short but still disclose:
- files changed
- validation run
- any skipped checks or assumptions

## Mandatory work products
For every non-trivial task, produce these three artifacts in the conversation, commit description, or review handoff.

### 1) Change schematic
A concise map of the intended change. Include:
- goal and affected behavior
- entrypoints and callers
- modules or layers touched
- data and state flow
- contracts and invariants to preserve
- side effects and trust boundaries
- rollback or recovery notes if stateful

### 2) Validation matrix
List:
- commands run
- commands intentionally not run
- why skipped commands were skipped
- residual risk from skipped validation

### 3) Gate status
For each applicable gate below, report one of:
- `PASS` — requirement met with evidence
- `WAIVED` — explicitly waived by the user or policy owner
- `BLOCKED` — missing evidence or unmet requirement

Do not hand off non-trivial coding work with an unacknowledged `BLOCKED` gate.

## Mandatory gates
Treat these as the default delivery state machine.

### G0 — Instruction gate
Required: always.

Pass when:
- the full applicable instruction chain has been read
- conflicting guidance has been resolved by precedence
- deeper local overrides are checked before editing

### G1 — Command gate
Required: always.

Pass when:
- canonical commands are identified from repository evidence
- guessed or substituted commands are not used
- validation scope is selected before code changes

Block when:
- commands are still guesses
- the validation plan relies on toolchain assumptions rather than repo evidence

### G2 — Schematic gate
Required: every non-trivial task.

Pass when a change schematic exists and answers:
- where the behavior starts
- what modules or layers change
- how data/state moves
- what contracts/invariants must stay true
- what side effects, failure modes, or rollback concerns exist

Block when:
- the change crosses layers without an explicit reason
- the intended ownership of logic or state is unclear
- contract changes are implicit rather than named

### G3 — Implementation gate
Required: every code change.

Pass when:
- the source of truth is changed instead of generated output
- the diff is minimal and coherent
- public behavior is preserved unless intentionally changed
- comments, names, and abstractions reflect domain intent
- unrelated refactors or formatting churn are excluded

Block when:
- generated files are hand-edited despite a generation path
- unrelated cleanup is mixed into the fix
- control flow, naming, or abstractions make the changed behavior harder to understand

### G4 — Validation gate
Required: every code change.

Pass when:
- relevant tests are added or updated where practical
- the smallest meaningful checks ran during iteration
- validation depth matches blast radius
- unrun checks are explicitly called out with residual risk

Block when:
- no evidence supports the claimed behavior
- failing checks are ignored or hidden
- assertions were weakened to make tests pass

### G5 — Clean-code gate
Required: every non-trivial task; strongly recommended for every code change.

Pass when the changed code is:
- cohesive: each changed unit has one obvious reason to change
- schematic: a reader can follow the top-level flow without hunting through indirection
- explicit: side effects, state transitions, and failure paths are visible
- low-friction: naming is precise, branching is controlled, nesting is shallow
- non-duplicative: repeated logic is removed or consciously justified
- typed/structured: avoid stringly-typed control flow, magic values, and boolean-flag APIs when better structure exists
- testable: seams are clear enough to validate behavior without fragile setup

Block when:
- a unit mixes parsing, policy, orchestration, I/O, and persistence with no clear separation
- hidden temporal coupling or ambient state is introduced
- complexity rises materially without necessity
- dead code, debug prints, broad catches, or TODO debt remain in changed paths without explicit tracking

### G6 — Security / privacy gate
Required: when trust boundaries, permissions, external input, network access, secrets, tenancy, or file handling change.

Pass when:
- changed trust boundaries are reviewed
- misuse cases are considered
- sensitive data handling remains safe
- validation covers the risky path

Block when:
- the change weakens authorization, validation, or data protection
- secrets or sensitive data could leak
- external input is trusted without adequate parsing, encoding, or verification

### G7 — Docs / contract gate
Required: when behavior, API, CLI, config, schema, examples, or migrations change.

Pass when:
- user-facing or contract-facing docs are updated
- examples and configuration references match the new behavior
- migration or rollout notes are included when needed

Block when:
- a behavior or contract changed but docs/examples/config did not

### G8 — Handoff gate
Required: every non-trivial task.

Pass when the final handoff includes:
- what changed
- why it changed
- change schematic
- commands run and outcomes
- commands not run
- gate status
- residual risk, waivers, or follow-ups

Block when:
- evidence is missing
- skipped validation is concealed
- material risk is not disclosed

### G9 — Instruction-eval gate
Required: when changing repository instructions, skills, or custom agents.

Pass when:
- routing behavior was checked with a small prompt set
- at least one negative trigger case was included
- any required output contract or schema was verified
- tool scope and least-privilege settings were reviewed where applicable

Block when:
- instructions changed with no routing or behavior check
- a skill description remains vague enough that triggering is unreliable
- a specialist gained broader tools than its role needs without justification

## Do this before changing code
1. Read the nearest applicable instruction chain.
2. Identify the canonical command set from CI and manifests.
3. Inspect adjacent implementation, tests, architecture notes, and any touched public contracts.
4. Decide whether the task is trivial or non-trivial.
5. For non-trivial work, produce a change schematic or ExecPlan before editing code.
6. Route through specialist subagents or matching skills, or emulate the same separation manually.

## Use subagents and skills for coding work
When the platform supports custom agents or subagents, use a team of specialists with narrow scope and least-privilege tools. If subagents are unavailable, emulate the same stages in one thread: plan, design, implement, validate, review, document, gate.

Expected companion profiles in `.github/agents/`:
- `feature-builder.agent.md` — top-level orchestrator and router
- `planner.agent.md` — read-only exploration and implementation plan
- `architecture-reviewer.agent.md` — read-only schematic and architecture-fit review
- `implementer.agent.md` — focused code changes
- `test-engineer.agent.md` — tests and validation
- `clean-code-reviewer.agent.md` — read-only maintainability and code-clarity review
- `reviewer.agent.md` — read-only regression, scope, and contract review
- `security-reviewer.agent.md` — read-only security review
- `docs-writer.agent.md` — docs, examples, migration notes, changelog
- `dependency-curator.agent.md` — dependency-change impact analysis
- `migration-planner.agent.md` — schema/state rollout and rollback planning
- `instruction-maintainer.agent.md` — AGENTS/skills/custom-agent maintenance and evaluation
- `gatekeeper.agent.md` — final gate evaluation before handoff

Expected repo skills:
- `exec-plan` — generate and refine a `PLANS.md` or equivalent for large work
- `ci-failure-triage` — summarize large CI/test failures and derive the minimal repro path
- `clean-code-gates` — run the schematic and maintainability checklist
- `migration-safety` — stage rollout, rollback, and compatibility checks for stateful changes
- `agent-instructions-eval` — validate changes to `AGENTS.md`, skills, and agent profiles

### Delegation rules
- Use **Planner** first for any non-trivial feature, refactor, migration, unfamiliar area, or unclear blast radius.
- Use **Architecture Reviewer** for every non-trivial task before implementation, and again after implementation if the design moved.
- Use **Implementer** only after the change shape is clear.
- Use **Test Engineer** for every bug fix, behavior change, flaky test, and regression risk.
- Use **Clean Code Reviewer** for every non-trivial change and every refactor.
- Use **Reviewer** before handoff on non-trivial changes to catch regressions, scope creep, and missing validation.
- Use **Security Reviewer** for auth, permissions, secrets, file I/O, networking, parsing, deserialization, uploads, templates, command execution, sandbox boundaries, or multi-tenant concerns.
- Use **Docs Writer** for externally visible behavior, API, CLI, config, onboarding, migration, or example changes.
- Use **Dependency Curator** before major library/framework/runtime upgrades or production dependency additions.
- Use **Migration Planner** before schema, state, data backfill, or rollout-sensitive changes.
- Use **Instruction Maintainer** when modifying `AGENTS.md`, skills, custom agents, eval prompts, or agent config.
- Use **Gatekeeper** as the final step for every non-trivial task.

### Concurrency and handoff rules
- Run read-only specialists in parallel when useful: `planner`, `architecture-reviewer`, `reviewer`, and `security-reviewer` can often work concurrently on discovery or final review.
- Only one write-capable specialist should mutate the codebase at a time.
- Internal specialists should usually be hidden from direct user selection via `user-invocable: false`; disable automatic invocation when explicit orchestration is preferred.
- Every handoff should carry: goal, files/areas, invariants, proposed commands, risks, and unresolved questions.
- Specialist output should be short and operational.

### Default routes
- **trivial edit** -> implementer -> targeted validation
- **bug fix** -> planner (if needed) -> implementer -> test-engineer -> reviewer
- **feature** -> exec-plan skill -> planner -> architecture-reviewer -> implementer -> test-engineer -> clean-code-reviewer -> reviewer -> docs-writer if needed -> gatekeeper
- **refactor** -> exec-plan skill -> planner -> architecture-reviewer -> implementer -> test-engineer -> clean-code-reviewer -> reviewer -> gatekeeper
- **dependency change** -> planner -> dependency-curator -> architecture-reviewer -> implementer -> test-engineer -> reviewer -> gatekeeper
- **migration / schema / state change** -> migration-safety skill -> planner -> migration-planner -> architecture-reviewer -> implementer -> test-engineer -> security-reviewer -> docs-writer -> clean-code-reviewer -> reviewer -> gatekeeper
- **failing CI / flaky test** -> ci-failure-triage skill -> planner or implementer -> test-engineer -> reviewer -> gatekeeper
- **AGENTS / skills / custom-agent change** -> instruction-maintainer -> agent-instructions-eval skill -> reviewer -> gatekeeper

## Coding rules by task type

### Bug fixes
- reproduce the bug or identify the failing contract first
- fix root cause, not just symptoms, unless a tactical patch is explicitly requested
- add or update regression coverage where practical
- avoid bundling opportunistic cleanup into the same diff

### Features
- preserve existing contracts unless the feature intentionally changes them
- make extension points explicit rather than sneaking in hidden coupling
- update docs/examples/config for any user-visible change
- prefer the smallest viable abstraction that fits current usage

### Refactors
- do not mix refactors with behavior changes unless explicitly required
- preserve behavior with tests or equivalence checks
- bias toward readability, ownership clarity, and removal of duplication
- stop when the changed area is materially clearer; do not boil the ocean

### Dependency changes
- prefer the smallest version or package change that solves the task
- assess runtime, build, type, test, and transitive impacts
- keep lockfile churn minimal and explain any large churn
- major upgrades require explicit compatibility notes and broader validation

### Migrations and stateful changes
- stage forward and rollback paths explicitly
- distinguish additive, compatibility-window, backfill, cutover, and cleanup steps
- call out destructive or irreversible operations
- never hide operational risk behind vague “migration required” wording

### Generated code and assets
- edit the source of truth, then regenerate
- do not hand-edit generated output unless the repository explicitly treats it as source
- include the regeneration command in the validation matrix
- separate generator changes from regenerated bulk output when practical

## Boundaries: always / ask first / never

### Always
- prefer minimal, reviewable diffs
- preserve backward compatibility unless intentionally changing it
- make state transitions, side effects, and error paths explicit
- keep names intention-revealing and control flow easy to follow
- disclose assumptions, skipped checks, and residual risk

### Ask first
- add production dependencies when policy requires approval
- change schemas, migrations, or stateful rollout behavior
- alter auth, permissions, billing, secrets, networking, or sandbox policy
- delete files, rename public modules, or change external contracts
- modify CI, deployment, or infra in repositories where these are approval-gated

### Never
- invent repository commands
- claim tests passed when they were not run
- hide failing checks or weaken tests to get green
- mix unrelated refactors into a fix
- hand-edit generated files when a generator exists
- leave debug prints, commented-out code, or silent broad catches in changed paths without explicit justification

## Sensitive paths and local overrides
Add real path-specific rules here. Keep them close to the code they govern.

Examples:
- `docs/**` — docs-only edits allowed; keep snippets runnable
- `generated/**` — generated; edit source specs or generators instead
- `migrations/**` — require rollout and rollback notes
- `infra/**` — high blast radius; broader validation required
- `public-api/**` — preserve backward compatibility unless explicitly changed
- `security/**` — require security review
- `examples/**` — examples must run or clearly state what is illustrative only

## Git and PR workflow
Replace with repository-specific conventions if they exist.

- Keep commits and PRs scoped to one coherent purpose.
- Mention the exact validation run in the PR body or handoff summary.
- Separate mechanical changes from behavioral changes when practical.
- If the repository uses changesets, release notes, conventional commits, or PR templates, follow them exactly.
- When parallel agent threads work on the same repository, use separate branches or git worktrees to avoid overlap.

## Keeping this file healthy
- Update this file when commands, paths, CI, or workflow rules change.
- Keep it specific, executable, and short enough that critical guidance is not truncated.
- Prefer concrete commands and real code examples over abstract prose.
- When the same corrective prompt is repeated twice, convert it into a skill, nested override, or specialist agent.
- When instructions grow, split by directory or workflow instead of appending more global text.

## Maintainer appendix: verification and tuning
Use these checks when maintaining the agent system itself.

### Verify instruction discovery
- Start the agent in the target subdirectory and ask it to list the instruction sources it loaded.
- Test both repository root and at least one deep subdirectory that has local overrides.

### Verify routing and trigger quality
- Use a small prompt set with both positive and negative trigger cases.
- Check whether the right skill or specialist is chosen.
- Check whether the expected commands or tools were used.
- Check whether outputs follow the intended schema or rubric when one exists.

### Tune for portability
- Keep repo skills in `.agents/skills/` for Codex-compatible setups.
- Mirror them in `.github/skills/` for GitHub Copilot compatibility when helpful.
- Keep GitHub custom-agent prompts focused and small; prefer more specialists over one huge profile.


### Stack
- Languages and versions: `Python 3.13` as the default stable runtime, `Python 3.14t` for high-throughput execution profiles, `C++17+` or `Rust stable` for performance-critical native components
- Frameworks/runtime: `Python asyncio` with a multi-pool executor model, `CuPy` and `PyTorch` for accelerated compute workloads
- Package/build tools: `uv` for Python environment and package management, `setuptools` for packaging, `maturin` or `cargo` for Rust-backed extensions, `cmake` for C++ native builds when needed
- Test stack: `pytest`, `pytest-asyncio`, `unittest` for compatibility cases, and targeted benchmark/perf smoke tests for native or GPU-heavy paths
- Linters/formatters: `Ruff` for linting and formatting
- Primary datastore(s): `SQLite` for local metadata/state by default, with file-based artifacts on disk for prompts, templates, and generated outputs
- Local service dependencies: `Docker Compose` when auxiliary services are needed, local CUDA runtime/toolkit for GPU execution, and optional Redis for queueing or coordination

### Path ownership
- `agentic_toolbelt/**` — source-owned package payload and installer logic; editable by maintainers, not generated
- `tests/**` — source-owned validation; changes require matching behavior updates and regression coverage
- `docs/**` — user-facing static documentation; validate after CLI or packaging changes
- `dist/**`, `build/**`, `*.egg-info/**`, `__pycache__/**` — generated artifacts; do not edit manually
- `pyproject.toml`, `setup.py` — packaging surface; special validation required because versioning, entry points, and package data must stay aligned
- `agentic_toolbelt/AGENTS.md` — high-impact instruction surface; never change behaviorally without explicit review of downstream impact

### Public contract surfaces
- CLI flags / commands: `agentic-set`, `agentic-set --force`
- HTTP / RPC / GraphQL / event schemas: `None currently; this project is a packaging/install tool rather than a network service`
- Database schema / migrations: `None currently; no formal database schema or migration system is required by default`
- Config files / env vars: `pyproject.toml`, `setup.py`, optional `CUDA_VISIBLE_DEVICES`, `PYTHONPATH`, and tool-specific env vars used by `uv`, PyTorch, or CuPy`
- Generated SDKs / clients / docs: `docs/index.html` is generated/maintained as the public docs surface; package artifacts under `dist/` are generated release outputs


## Optional nested files to add in real repositories
Use deeper `AGENTS.md` or `AGENTS.override.md` files when local rules differ materially from the root file.

Examples:
- `docs/AGENTS.md` — docs-only writing rules and docs validation commands
- `migrations/AGENTS.override.md` — rollout/rollback and approval rules
- `packages/mobile/AGENTS.md` — platform-specific build/test commands
- `services/payments/AGENTS.override.md` — stricter trust-boundary and validation rules

Keep local overrides narrow. The closer file should only add or replace what differs.



<!-- FAST-TOOLS PROMPT v1 | codex-mastery | watermark:do-not-alter -->

## CRITICAL: Use ripgrep, not grep

NEVER use grep for project-wide searches (slow, ignores .gitignore). ALWAYS use rg.

- `rg "pattern"` — search content
- `rg --files | rg "name"` — find files
- `rg -t python "def"` — language filters

## File finding

- Prefer `fd` (or `fdfind` on Debian/Ubuntu). Respects .gitignore.

## JSON

- Use `jq` for parsing and transformations.

## Install Guidance

- macOS: `brew install ripgrep fd jq`
- Debian/Ubuntu: `sudo apt update && sudo apt install -y ripgrep fd-find jq` (alias `fd=fdfind`)

## Agent Instructions

- Replace commands: grep→rg, find→rg --files/fd, ls -R→rg --files, cat|grep→rg pattern file
- Cap reads at 250 lines; prefer `rg -n -A 3 -B 3` for context
- Use `jq` for JSON instead of regex

<!-- END FAST-TOOLS PROMPT v1 | codex-mastery -->
