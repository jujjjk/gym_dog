"""Keep mocked controller tests separate from the real hardware owner lock."""
import builtins
import os
import pytest


@pytest.fixture(autouse=True)
def isolated_mock_controller_lock(monkeypatch, tmp_path):
    original = builtins.open
    def open_test_file(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and os.fspath(file) == '/tmp/mydog_a6850_controller.lock':
            file = tmp_path / 'mock_controller.lock'
        return original(file, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', open_test_file)
