# ADR-0003: Logging 

 
The CLI and library need predictable operational logging with user controls and stable conventions for module loggers.

## Decision

- Use `logging.getLogger(__name__)` in modules.
- Configure stdlib logging in CLI entrypoint.
- CLI controls:
  - `-v/--verbose` (repeatable)
  - `-q/--quiet` (repeatable)
  - default level: `INFO`
  - `--log-level {DEBUG,INFO,WARNING,ERROR}` for explicit override
- Routing policy for CLI:
  - `DEBUG/INFO` -> stdout
  - `WARNING/ERROR` -> stderr

## Consequences

- Logs are consistent with Python stdlib patterns.
- Operators can control verbosity without code changes.
- CLI behavior is script-friendly and easier to integrate in pipelines.

