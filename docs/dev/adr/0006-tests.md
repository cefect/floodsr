# testing
- use pytest
- each module should have a corresponding module in tests
- test data should be in `tests/data` and small enough for quick unit tests
- test structure should mirror module structure for discoverability
- tests should minimize extra code and focus on input/output contract validation
- use fixtures for common test data setup (`tests/conftest.py`)