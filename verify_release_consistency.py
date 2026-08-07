#!/usr/bin/env python3
"""Static verifier for release consistency.

This checker parses source text/AST without importing heavy scientific
libraries. It verifies that shared audit primitives live in `audit_common.py`,
that experiment scripts do not rebuild kNN/shortest-path graphs locally, and
that formal decision assignments do not use a p-value OR diagnostic fallback.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
SCRIPTS = sorted(SCRIPTS_DIR.glob("*.py")) if SCRIPTS_DIR.exists() else []
REQUIRED_HEAVY_SCRIPTS = {
    "code_10_neural_trajectory_pipeline.py",
    "code_14_neural_fullspace_audit.py",
    "code_15_resnet_threshold_sensitivity.py",
    "code_16_resnet_positive_control_lambda0.py",
    "code_17_resnet_positive_control_lambda1.py",
    "code_18_step_acf_analysis.py",
    "code_19_ar_block_calibration.py",
    "code_20_graph_locality_diagnostics.py",
}

CORE_NAMES = {
    "compute_stats",
    "geodesic_distance_matrix",
    "matched_step_null",
    "matched_step_nulls",
    "block_null",
    "block_step_nulls",
    "safe_lifetime",
    "sha256_file",
    "distance_from_step_gram",
    "step_lengths_from_gram",
    "assert_gram_reconstruction",
}
GRAPH_CALL_NAMES = {"NearestNeighbors", "kneighbors_graph", "shortest_path"}
LOCAL_GRAM_DISTANCE_NAMES = {
    "distance_from_step_gram",
    "_distance_from_step_gram",
    "cumulative_distance_from_step_gram",
}
SUSPICIOUS_DECISION_NAMES = {
    "formal_trigger",
    "formal",
    "decision",
    "gatekeeper",
    "recurrence",
    "is_recurrence",
    "detected",
}

errors: list[str] = []
warnings_out: list[str] = []


def read_tree(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        errors.append(f"{path.relative_to(ROOT)} has SyntaxError at line {exc.lineno}: {exc.msg}")
        return ast.Module(body=[], type_ignores=[])


def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def target_names(target: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names |= target_names(elt)
    elif isinstance(target, ast.Attribute):
        names.add(target.attr)
    elif isinstance(target, ast.Subscript):
        names |= target_names(target.value)
    return names


def contains_name(node: ast.AST, names: set[str]) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in names:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in names:
            return True
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value in names:
            return True
    return False


def check_sentinel_default(common_tree: ast.Module) -> None:
    for node in common_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"geodesic_distance_matrix", "geodesic_h1_lifetime"}:
            defaults = node.args.defaults
            kw_names = [a.arg for a in node.args.kwonlyargs]
            kw_defaults = dict(zip(kw_names, node.args.kw_defaults))
            default = kw_defaults.get("sentinel_fill")
            if not (isinstance(default, ast.Constant) and default.value is False):
                errors.append(
                    f"audit_common.py:{node.name} must default sentinel_fill=False for reviewer-safe formal runs"
                )


def check_or_gate(path: Path, tree: ast.Module) -> None:
    """Flag formal recurrence logic that combines p-values with diagnostics."""
    rel = path.relative_to(ROOT)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            assigned = set().union(*(target_names(t) for t in node.targets))
            assigned_lower = {name.lower() for name in assigned}
            is_decision_target = bool(assigned_lower & SUSPICIOUS_DECISION_NAMES)
            if isinstance(node.value, ast.BoolOp) and isinstance(node.value.op, ast.Or):
                mentions_p = contains_name(node.value, {"pperm", "p_perm", "p_value", "pvalue", "perm_p"})
                mentions_diag = contains_name(node.value, {"fallback", "separation_flag", "nominal_trigger"})
                if is_decision_target and mentions_p and mentions_diag:
                    errors.append(
                        f"{rel} line {node.lineno}: possible p-value OR diagnostic-fallback gate in formal decision assignment"
                    )
        elif isinstance(node, ast.AnnAssign):
            assigned = target_names(node.target)
            assigned_lower = {name.lower() for name in assigned}
            if node.value is not None and isinstance(node.value, ast.BoolOp) and isinstance(node.value.op, ast.Or):
                mentions_p = contains_name(node.value, {"pperm", "p_perm", "p_value", "pvalue", "perm_p"})
                mentions_diag = contains_name(node.value, {"fallback", "separation_flag", "nominal_trigger"})
                if assigned_lower & SUSPICIOUS_DECISION_NAMES and mentions_p and mentions_diag:
                    errors.append(
                        f"{rel} line {node.lineno}: possible p-value OR diagnostic-fallback gate in formal decision assignment"
                    )

    text = path.read_text(encoding="utf-8")
    if "nominal_trigger" in text:
        warnings_out.append(
            f"{rel}: contains nominal_trigger; verify it is treated as diagnostic only, never formal recurrence"
        )


def main() -> None:
    common_path = ROOT / "audit_common.py"
    if not common_path.exists():
        errors.append("audit_common.py is missing from repository root")
        common_tree = ast.Module(body=[], type_ignores=[])
    else:
        common_tree = read_tree(common_path)
        common_defs = {n.name for n in common_tree.body if isinstance(n, ast.FunctionDef)}
        missing = CORE_NAMES - common_defs
        if missing:
            errors.append(f"audit_common.py missing core definitions: {sorted(missing)}")
        check_sentinel_default(common_tree)

    if not SCRIPTS_DIR.exists():
        errors.append("scripts/ directory is missing; cannot verify experiment shared-logic consistency")
    elif not SCRIPTS:
        errors.append("scripts/ directory contains no .py files; cannot verify experiment shared-logic consistency")
    else:
        present = {path.name for path in SCRIPTS}
        missing_heavy = sorted(REQUIRED_HEAVY_SCRIPTS - present)
        if missing_heavy:
            errors.append(f"scripts/ is missing required heavy-run source: {missing_heavy}")

    for path in SCRIPTS:
        tree = read_tree(path)
        rel = path.relative_to(ROOT)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in CORE_NAMES:
                errors.append(f"{rel} defines centralized primitive {node.name} at line {node.lineno}")
            if isinstance(node, ast.FunctionDef) and node.name in LOCAL_GRAM_DISTANCE_NAMES:
                errors.append(
                    f"{rel} defines local increment-Gram distance reconstruction "
                    f"{node.name} at line {node.lineno}; import audit_common instead"
                )
            if isinstance(node, ast.Call):
                name = call_name(node.func)
                if name in GRAPH_CALL_NAMES:
                    errors.append(f"{rel} directly calls graph builder {name} at line {node.lineno}")
        check_or_gate(path, tree)

    # code_21 lives at repository root rather than scripts/, but it is a
    # production Phase 1 component and must use the same shared distance path.
    certificate_path = ROOT / "code_21_path_fidelity_certificate.py"
    if not certificate_path.exists():
        errors.append("code_21_path_fidelity_certificate.py is missing from repository root")
    else:
        certificate_tree = read_tree(certificate_path)
        for node in ast.walk(certificate_tree):
            if isinstance(node, ast.FunctionDef) and node.name in LOCAL_GRAM_DISTANCE_NAMES:
                errors.append(
                    "code_21_path_fidelity_certificate.py defines a production "
                    f"increment-Gram distance reconstruction at line {node.lineno}"
                )
            if isinstance(node, ast.Call) and call_name(node.func) == "_legacy_distance_from_step_gram_prefix":
                errors.append(
                    "code_21_path_fidelity_certificate.py calls the legacy prefix reconstruction"
                )
        certificate_text = certificate_path.read_text(encoding="utf-8")
        if "shared_distance_from_step_gram" not in certificate_text:
            errors.append("code_21_path_fidelity_certificate.py does not import the shared distance routine")

    manifest_path = ROOT / "make_manifests.py"
    if manifest_path.exists():
        mtree = read_tree(manifest_path)
        for node in mtree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "sha256_file":
                errors.append("make_manifests.py defines sha256_file instead of importing audit_common.sha256_file")
    else:
        warnings_out.append("make_manifests.py is missing; release checksums cannot be generated")

    runner_path = ROOT / "reproduce_all.sh"
    if not runner_path.exists():
        errors.append("reproduce_all.sh is missing")
    else:
        runner_text = runner_path.read_text(encoding="utf-8")
        if "rm -rf results figures tables manifests logs acf_results exp5_results" in runner_text:
            errors.append("reproduce_all.sh still deletes exp5_results and its trajectory archive during CLEAN")
        if "code_15_resnet_threshold_sensitivity.py" not in runner_text:
            errors.append("reproduce_all.sh does not run restored code_15_resnet_threshold_sensitivity.py")
        if "code_21_path_fidelity_certificate.py" not in runner_text:
            errors.append("reproduce_all.sh does not run the Phase 1 certificate")

    if warnings_out:
        print("Release consistency warnings:")
        for warn in warnings_out:
            print(f"  - {warn}")

    if errors:
        print("Release consistency check FAILED:")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)

    print("Release consistency check passed.")
    print("Core primitives are owned by audit_common.py; scripts do not directly call kNN/shortest-path graph builders.")
    print("All production increment-Gram distance reconstruction is centralised in audit_common.py.")
    print("No formal p-value OR diagnostic-fallback gate was detected in script decision assignments.")


if __name__ == "__main__":
    main()
