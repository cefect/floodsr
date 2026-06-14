# ADR-0019: General Code Style

- include comments so the flow is readable. prefer comments at phase boundaries, not line-by-line narration
- avoid passing *args to functions unless strictly necessary
- prefer explicit signatures over **kwargs unless there is a real extensibility need.
- prefer assertions and small boundary checks near the top of functions