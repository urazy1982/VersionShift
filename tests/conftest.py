"""Public package omits private byte-authoritative E2 metadata and personal approval."""
import pytest

def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.name == "test_separate_author_gate.py":
            item.add_marker(pytest.mark.skip(reason="Controlled host-only gate requires byte-exact E2 containing excluded personal path; public view deliberately has another digest."))
