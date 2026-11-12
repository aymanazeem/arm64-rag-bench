# Repository and development setup

## Current state

The repository was bootstrapped on `chore/project-bootstrap`. An empty commit was created on this branch, and `main` initially pointed to that empty commit. Project files were prepared on the setup branch for introduction through the first PR.

Remote: [aymanazeem/arm64-rag-bench](https://github.com/aymanazeem/arm64-rag-bench), now public. Personal authentication is connected. Initial files entered `main` through [PR #1](https://github.com/aymanazeem/arm64-rag-bench/pull/1).

Initial setup checks passed, including 22 Git-hook cases. The [runtime guide](runtime.md) covers the current environment and tests.

Only this repository's Git identity is configured. The existing global work identity is not changed. The personal GitHub profile ID was checked through GitHub's API; the repository uses `110621518+aymanazeem@users.noreply.github.com` to avoid publishing a work or personal email address.

## Personal GitHub connection

Authenticate from a terminal using the personal account in the browser:

```sh
gh auth login --hostname github.com --git-protocol https --web
gh api user --jq .login
```

The second command must print `aymanazeem` before repository creation. If both accounts are stored, `gh auth switch --hostname github.com --user aymanazeem` selects the personal account. Authentication selection can affect other CLI sessions; it does not change Git commit identity. Do not remove the existing work account or put tokens in files or chat.

## First publication

The original private-repository bootstrap was:

1. Check whether `aymanazeem/arm64-rag-bench` already exists. Reuse it only if it is the intended private repository; do not overwrite unrelated history.
2. Create the private repository without generated README, license or other competing history, and add its URL as `origin`.
3. Verify owner, visibility and the empty remote before sending any files.
4. Publish the empty base as `main`, then publish `chore/project-bootstrap`. No project file is committed directly on `main`.
5. Set the default branch to `main`; open the setup PR, perform a review and merge only after its checks pass. Use a merge strategy and commit message that preserve the owner's identity and omit co-author trailers. Inspect the resulting commit identity; do not claim GitHub platform metadata is configurable in every merge method.
6. Enable server-side PR and status-check requirements if supported by the account's private-repository plan. Disable force pushes and deletion of `main`. Do not require a second human approval in this solo workflow; a PR author cannot approve their own PR. Report any unavailable enforcement rather than claiming protection is active.

Normal changes are committed and pushed on feature branches, reviewed, then merged through PRs. The local pre-push hook blocks updates to existing `main` and `master`; its sole initialization allowance is creating an absent `main` at the one empty root commit.

The owner made the repository public on 2026-11-12. Branch protection requires PRs, resolved review conversations and the successful GitHub Actions `checks` job, including for administrators. Branches must be up to date before merge. Force pushes and branch deletion are blocked. Reviews are recorded without requiring a second person's approval.

## Codex and local safeguards

Working rules (`AGENTS.md`), plans (`PLANS.md`) and agent settings (`.codex/`, `.agents/`) stay local and are ignored by Git. The commit hook also blocks them if force-added. They are not included in a fresh clone.

Local configuration can set reasoning effort without changing permissions or storing credentials. Project configuration loads only for trusted projects; managed client policies still apply.

The tracked Git hooks refuse commits on `main`/`master`, commits using another identity, and co-author/generated-by commit attribution. A pre-push hook refuses ordinary direct updates to protected branches. These are local safeguards and must be enabled in each clone; they are not a substitute for GitHub branch rules. PR descriptions also need review for the attribution rule.

No extra project skill or duplicated instruction file is needed yet. Add focused skills only when a real repeated workflow warrants them.

Official references: [repository instructions](https://learn.chatgpt.com/docs/agent-configuration/agents-md), [project configuration](https://learn.chatgpt.com/docs/config-file/config-basic), [configuration keys](https://learn.chatgpt.com/docs/config-file/config-reference), [GitHub CLI login](https://cli.github.com/manual/gh_auth_login), [private repository creation](https://cli.github.com/manual/gh_repo_create).

## Observed development environment

Inventory checked on 2026-11-12:

| Item | Observed value |
| --- | --- |
| Hardware | MacBookPro18,2, Apple M1 Max |
| CPU | 10 cores: 8 performance, 2 efficiency |
| CPU instruction features | DotProd supported; I8MM and SME unsupported (queried with sysctl) |
| Unified memory | 32 GB |
| macOS | 26.7.1, build 25G241 |
| Compiler | Apple clang 21.0.0 |
| Command Line Tools | Installed |
| Git / GitHub CLI | Installed |
| uv | 0.12.23 |
| System Python | 3.9.6; use a project-managed version for the application |
| CMake / Ninja | Not found on the current PATH |
| Free disk | Approximately 700 GiB at inspection |

The runtime environment now uses Python 3.12.15, CMake 4.4.4 and Ninja 1.13.2 through uv. System Python is unchanged. See the [current inventory](results/runtime-feasibility/hardware.json). No paid account or hosted model API is required.
