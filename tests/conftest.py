"""
Shared fixtures and test configuration.
Prevents real SignalK connections and API calls during tests.
"""
import os
import pytest

# Block any real external calls during tests
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")
os.environ.setdefault("SIGNALK_HOST", "localhost")
os.environ.setdefault("SIGNALK_PORT", "3000")
