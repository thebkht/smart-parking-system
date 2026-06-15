# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities.

Instead, report them privately through GitHub's
[private vulnerability reporting](https://github.com/thebkht/smart-parking-system/security/advisories/new),
or email the maintainer at **me@thebkht.com**.

Please include:

- a description of the issue and its impact,
- steps to reproduce or a proof of concept,
- affected component (edge, backend, web, mobile, ml) and version/commit.

We will acknowledge your report and work with you on a fix and disclosure
timeline.

## Scope and notes

- **Authentication is opt-in.** The backend runs **unauthenticated by default**;
  bearer-token auth on owner routes is enabled only when `AUTH_ENABLED=1`. The
  default configuration is intended for local development and demos, not for
  exposing the backend to untrusted networks.
- Do not test against deployments you do not own or have permission to test.
- The SQLite database, datasets, and model weights are not part of the
  repository and are out of scope for source-level reports.
