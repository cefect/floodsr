# ADR-M-0000: General Model Worker Style

- add numerical prefixes to the methods so the major phases are readable (e.g, _1pre-process)
- use class objects with methods to organize model-specific logic rather than standalone functions
-  model workers should have a tight shared run(...) contract and should not silently swallow unexpected kwargs.
- workers should validate the shared prepared-raster boundary and not re-own CRS/alignment policy unless the model truly needs it.
- every worker should return the same pattern, including structured runtime metadata, including effective model-specific params