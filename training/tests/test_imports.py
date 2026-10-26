"""Import every shipped training module to catch incomplete source exports."""
import importlib
from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODULES = sorted(".".join(p.relative_to(ROOT).with_suffix("").parts) for folder in ["src", "scripts", "configs"] for p in (ROOT / folder).rglob("*.py") if p.name != "__init__.py")
@pytest.mark.parametrize("module", MODULES)
def test_import(module, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module])
    importlib.import_module(module)
