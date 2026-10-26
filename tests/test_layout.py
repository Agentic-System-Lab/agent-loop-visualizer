"""The two packages: agent/ runs a turn and app/ is what a person sees, and the agent never knows the app.

app/ imports agent/, never the other way round: the agent says what happens only as events, so it runs without the
page or Streamlit (in these tests, in pi's endpoint, in another app). This reads every file of agent/ for an import
of either, at the top or inside a function, rather than trusting a reviewer to spot one.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
AGENT_FILES = sorted((ROOT / "agent").rglob("*.py"))
APP_ONLY = ("app", "streamlit")  # what only the page may import


def imported(path: Path) -> list[tuple[int, str]]:
    """Each module the file imports, with its line: `import a.b` and `from a.b import c` both name a.b. A relative
    import stays inside agent/, so it is left out."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(path))):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
    return found


def test_the_check_sees_every_way_of_importing(tmp_path):
    module = tmp_path / "module.py"
    module.write_text("import streamlit as st\nfrom app import render\nfrom app.page import main\n"
                      "def later():\n    import app.guide\nfrom . import tools\nimport application\n")
    assert sorted(name for _, name in imported(module)) == ["app", "app.guide", "app.page", "application", "streamlit"]
    assert sorted(name for _, name in imported(module) if name.split(".")[0] in APP_ONLY) == ["app", "app.guide", "app.page", "streamlit"]


@pytest.mark.parametrize("path", AGENT_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_the_agent_never_imports_the_app_or_streamlit(path):
    found = [f"{path.relative_to(ROOT)}:{line} imports {name}" for line, name in imported(path) if name.split(".")[0] in APP_ONLY]
    assert not found, found


def test_every_module_of_the_agent_is_read():
    assert {p.relative_to(ROOT).parts[1] for p in AGENT_FILES} >= {"harnesses", "model_backends", "model_families", "tools.py", "media.py"}
