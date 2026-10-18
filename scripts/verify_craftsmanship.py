"""Automated craftsmanship, zero-mocking, and testing pyramid verifier for future-music."""

import ast
import sys
from pathlib import Path

MAX_METHOD_LINES = 20
MAX_CLASS_LINES = 120
MAX_COMPLEXITY = 3
FORBIDDEN_DOMAIN_IMPORTS = {
    "fastapi",
    "uvicorn",
    "mido",
    "httpx",
    "application",
    "infrastructure",
    "ports",
    "sounddevice",
    "starlette",
    "pydantic",
    "random",
}
FORBIDDEN_MOCK_SYMBOLS = {"mock", "MagicMock", "AsyncMock", "patch", "mocker"}
BRANCH_TYPES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.IfExp,
    ast.comprehension,
)
FuncNode = ast.FunctionDef | ast.AsyncFunctionDef


def count_branch_nodes(node: ast.AST) -> int:
    """Count AST branch points that add to McCabe cyclomatic complexity."""
    branches = sum(1 for child in ast.walk(node) if isinstance(child, BRANCH_TYPES))
    bool_ops = sum(
        len(child.values) - 1 for child in ast.walk(node) if isinstance(child, ast.BoolOp)
    )
    return 1 + branches + bool_ops


def check_function_metrics(filepath: Path, node: FuncNode) -> list[str]:
    """Verify function length and cyclomatic complexity."""
    errors: list[str] = []
    span = (node.end_lineno or node.lineno) - node.lineno + 1
    comp = count_branch_nodes(node)
    loc = f"{filepath}:{node.lineno} function '{node.name}'"
    if span > MAX_METHOD_LINES:
        errors.append(f"{loc} is {span} lines (max {MAX_METHOD_LINES})")
    if comp > MAX_COMPLEXITY:
        errors.append(f"{loc} has complexity {comp} (max {MAX_COMPLEXITY})")
    return errors


def check_class_metrics(filepath: Path, node: ast.ClassDef) -> list[str]:
    """Verify class line length."""
    span = (node.end_lineno or node.lineno) - node.lineno + 1
    if span > MAX_CLASS_LINES:
        msg = (
            f"{filepath}:{node.lineno} class '{node.name}' is {span} lines (max {MAX_CLASS_LINES})"
        )
        return [msg]
    return []


def tokens_from_import_node(node: ast.AST) -> set[str]:
    """Extract module name tokens from a single Import or ImportFrom node."""
    if isinstance(node, ast.Import):
        return {part for alias in node.names for part in alias.name.split(".")}
    if isinstance(node, ast.ImportFrom) and node.module:
        return set(node.module.split("."))
    return set()


def extract_import_tokens(tree: ast.AST) -> set[str]:
    """Extract all module path tokens imported in an AST."""
    tokens: set[str] = set()
    for node in ast.walk(tree):
        tokens.update(tokens_from_import_node(node))
    return tokens


def check_source_imports(filepath: Path, tree: ast.AST) -> list[str]:
    """Ensure domain purity and zero random usage in production source."""
    imported = extract_import_tokens(tree)
    errors: list[str] = []
    if "random" in imported:
        errors.append(f"{filepath}: forbidden 'random' import in [VERIFIED ENGINE] source")
    if "domain" in filepath.parts and (imported & FORBIDDEN_DOMAIN_IMPORTS):
        bad = sorted(imported & FORBIDDEN_DOMAIN_IMPORTS)
        errors.append(f"{filepath}: domain purity violation importing {bad}")
    return errors


def inspect_ast_node(filepath: Path, node: ast.AST) -> list[str]:
    """Check a single AST node for function or class metric violations."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return check_function_metrics(filepath, node)
    if isinstance(node, ast.ClassDef):
        return check_class_metrics(filepath, node)
    return []


def inspect_source_file(filepath: Path) -> list[str]:
    """Run all AST craftsmanship checks on a single source file."""
    tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
    errors = check_source_imports(filepath, tree)
    for node in ast.walk(tree):
        errors.extend(inspect_ast_node(filepath, node))
    return errors


def is_nontrivial_assert(node: ast.AST) -> bool:
    """Return True if node is a non-trivial assert or pytest.raises reference."""
    if isinstance(node, ast.Assert):
        return not (isinstance(node.test, ast.Constant) and node.test.value is True)
    return isinstance(node, ast.Attribute) and node.attr == "raises"


def has_substantive_assertion(func_node: FuncNode) -> bool:
    """Return True if a test function has a real assert or pytest.raises."""
    return any(is_nontrivial_assert(n) for n in ast.walk(func_node))


def find_test_functions(tree: ast.AST) -> list[FuncNode]:
    """Return all test_* function definitions in an AST."""
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
    ]


def inspect_test_file(filepath: Path) -> list[str]:
    """Verify zero mocking and substantive assertions in a test file."""
    tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
    names = extract_import_tokens(tree) | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    errors: list[str] = []
    if names & FORBIDDEN_MOCK_SYMBOLS:
        bad = sorted(names & FORBIDDEN_MOCK_SYMBOLS)
        errors.append(f"{filepath}: forbidden mocking symbol(s) detected: {bad}")
    missing = [fn for fn in find_test_functions(tree) if not has_substantive_assertion(fn)]
    errors.extend(
        f"{filepath}:{fn.lineno} test '{fn.name}' lacks a substantive assertion" for fn in missing
    )
    return errors


def count_tests_in_dir(directory: Path) -> int:
    """Count test_* functions inside a test directory."""
    if not directory.exists():
        return 0
    return sum(
        len(find_test_functions(ast.parse(p.read_text(encoding="utf-8"))))
        for p in directory.rglob("test_*.py")
    )


def verify_testing_pyramid(tests_root: Path) -> list[str]:
    """Ensure unit > integration >= acceptance >= 1 whenever tests exist."""
    u = count_tests_in_dir(tests_root / "unit")
    i = count_tests_in_dir(tests_root / "integration")
    a = count_tests_in_dir(tests_root / "acceptance")
    print(f"Testing Pyramid: unit={u}, integration={i}, acceptance={a}")
    if (u + i + a) > 0 and not (u > i >= a >= 1):
        return [
            f"Testing pyramid violation: unit={u}, integration={i}, acceptance={a}. "
            "Must satisfy unit > integration >= acceptance >= 1."
        ]
    return []


def collect_test_files(tests_root: Path) -> list[Path]:
    """Return all test_*.py files under tests_root if it exists."""
    return list(tests_root.rglob("test_*.py")) if tests_root.exists() else []


def main() -> int:
    """Entry point for craftsmanship verification."""
    root = Path(__file__).resolve().parent.parent
    errors = [err for p in (root / "src").rglob("*.py") for err in inspect_source_file(p)]
    errors.extend(err for p in collect_test_files(root / "tests") for err in inspect_test_file(p))
    errors.extend(verify_testing_pyramid(root / "tests"))
    for err in errors:
        print(f"ERROR: {err}", file=sys.stderr)
    if not errors:
        print("Craftsmanship verification passed: all metrics, DDD, and pyramid checks green.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
