# Security policy

## Reporting a security vulnerability

Report security bugs privately. Do not report them through public GitHub issues, discussions or pull requests.

Use [GitHub Security Advisories](https://github.com/ontogr/sage-xgb/security/advisories/new) for the report. This uses the private vulnerability reporting feature of GitHub. You can also contact a maintainer through their GitHub profile.

Include as much of the following as you can, so that we can triage the report.

- A description of the vulnerability and its possible impact.
- Steps to reproduce the problem, or a proof of concept.
- The affected version or versions, if you know them.

We acknowledge each report as soon as we can, and we keep you informed while we work on a fix.

## Handling credentials

Kaggle credentials and the MLflow tracking URI belong in a local `.env` file, which Git ignores. The Docker build excludes `.env`. Do not commit a `.env` file, an API key or a token.
