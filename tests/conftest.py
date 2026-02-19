
import os, logging, sys, hashlib, json
import pytest, yaml
import pathlib
import pandas as pd
 
 
 

# project parametesr
# Get the project root (parent of tests directory)
project_root = pathlib.Path(__file__).parent.parent
 

 
#===============================================================================
# pytest custom config------------
#===============================================================================
 

def pytest_runtest_teardown(item, nextitem):
    """custom teardown message"""
    test_name = item.name
    print(f"\n{'='*20} Test completed: {test_name} {'='*20}\n\n\n")
    
def pytest_report_header(config):
    """modifies the pytest header to show all of the arguments"""
    return f"pytest arguments: {' '.join(config.invocation_params.args)}"


# -------------------
# ----- Fixtures -----
# -------------------
@pytest.fixture(scope='session')
def logger():
    """Simple logger fixture for the function under test."""
    log = logging.getLogger("pytest")
    log.setLevel(logging.DEBUG)
    # keep handlers minimal to avoid duplicate logs across runs
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log

 