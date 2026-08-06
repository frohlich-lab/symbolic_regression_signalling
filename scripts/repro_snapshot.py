#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import importlib.metadata as imd
except Exception:  # pragma: no cover - importlib.metadata always exists in py3.8+
    imd = None  # type: ignore


PACKAGE_SPECS = {
    "python": [],
    "julia": [],
    "snakemake": [],
    "pysr": ["pysr"],
    "sympy": ["sympy"],
    "pysb": ["pysb"],
    "amici": ["amici"],
    "diffrax": ["diffrax"],
    "jax": ["jax", "jaxlib"],
    "optuna": ["optuna"],
    "scikit-learn": ["scikit-learn"],
    "torch": ["torch"],
    "tensorflow": ["tensorflow", "tensorflow-macos", "tensorflow-cpu"],
    "pykan": ["pykan", "kan"],
    "aifeynman": ["aifeynman", "AI-Feynman", "aifeynman2"],
    "dso": ["dso", "deep-symbolic-optimization"],
    "numpy": ["numpy"],
    "pandas": ["pandas"],
}

PACKAGE_ALIASES = {
    "pysr": ["pysr"],
    "sympy": ["sympy"],
    "pysb": ["pysb"],
    "amici": ["amici"],
    "diffrax": ["diffrax"],
    "jax": ["jax", "jaxlib"],
    "optuna": ["optuna"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "torch": ["torch", "pytorch"],
    "tensorflow": ["tensorflow", "tensorflow-macos", "tensorflow-cpu"],
    "pykan": ["pykan", "kan"],
    "aifeynman": ["aifeynman", "ai-feynman", "AI-Feynman", "aifeynman2"],
    "dso": ["dso", "deep-symbolic-optimization"],
    "numpy": ["numpy"],
    "pandas": ["pandas"],
    "snakemake": ["snakemake"],
    "julia": ["julia"],
}


def _run_cmd(cmd: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    return out if out else None


def _run_cmd_full(cmd: list[str], cwd: Path) -> dict:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {"error": "not_found", "command": cmd}
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _get_dist_version(names: list[str]) -> str | None:
    if imd is None:
        return None
    for name in names:
        try:
            return imd.version(name)
        except imd.PackageNotFoundError:
            continue
        except Exception:
            continue
    return None


def _collect_versions(root: Path) -> dict:
    versions: dict[str, str | None] = {}
    versions["python"] = sys.version.replace("\n", " ")
    versions["python_executable"] = sys.executable
    versions["julia"] = _run_cmd(["julia", "--version"], root)
    versions["snakemake"] = _run_cmd(["snakemake", "--version"], root)
    versions["git"] = _run_cmd(["git", "--version"], root)
    versions["pip"] = _run_cmd([sys.executable, "-m", "pip", "--version"], root)

    packages: dict[str, str | None] = {}
    for key, names in PACKAGE_SPECS.items():
        if key in ("python", "julia", "snakemake"):
            continue
        if not names:
            packages[key] = None
            continue
        packages[key] = _get_dist_version(names)
    versions["packages"] = packages
    return versions


def _is_relevant_env_file(path: Path, size_bytes: int | None = None) -> bool:
    path_str = str(path).lower()
    if "site-packages" in path_str or "node_modules" in path_str:
        return False
    if "jupyterlab" in path_str and "yarn.lock" in path_str:
        return False
    if size_bytes is not None and size_bytes > 2_000_000:
        return False
    name = path.name.lower()
    if any(token in name for token in ("requirements", "environment", "conda", "pipfile", "poetry.lock")):
        return True
    if "wandb" in path_str and "/files/" in path_str:
        return True
    if path.suffix.lower() in {".yml", ".yaml", ".txt", ".lock"} and "env" in path_str:
        return True
    return False


def _parse_env_specs(path: Path) -> list[dict]:
    specs = []
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return specs
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw.startswith("- "):
            raw = raw[2:].strip()
        if raw.lower() in {"pip:", "channels:", "dependencies:", "name:"}:
            continue
        if ":" in raw and not re.search(r"[<>=~]", raw):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*([=<>!~]{1,2})\s*([^\s#;]+)", raw)
        if match:
            specs.append(
                {
                    "name": match.group(1),
                    "op": match.group(2),
                    "version": match.group(3),
                    "raw": line.strip(),
                }
            )
            continue
        if "=" in raw and "==" not in raw:
            parts = raw.split("=")
            if len(parts) >= 2 and parts[0]:
                specs.append(
                    {
                        "name": parts[0],
                        "op": "=",
                        "version": "=".join(parts[1:]),
                        "raw": line.strip(),
                    }
                )
    return specs


def _collect_versions_from_env_files(root: Path, env_files: list[dict]) -> dict:
    packages: dict[str, list[dict]] = {k: [] for k in PACKAGE_ALIASES}
    for entry in env_files:
        rel_path = entry.get("path")
        if not rel_path:
            continue
        path = root / rel_path
        size = entry.get("bytes")
        if not _is_relevant_env_file(path, size):
            continue
        for spec in _parse_env_specs(path):
            name = spec["name"].lower()
            for key, aliases in PACKAGE_ALIASES.items():
                if name in [a.lower() for a in aliases]:
                    packages[key].append(
                        {
                            "path": rel_path,
                            "spec": spec["raw"],
                            "version": spec["version"],
                        }
                    )
    return packages


def _collect_git(root: Path) -> dict:
    git_dir = root / ".git"
    if not git_dir.exists():
        return {"present": False}
    commit = _run_cmd(["git", "rev-parse", "HEAD"], root)
    status = _run_cmd(["git", "status", "--porcelain"], root)
    dirty_files = []
    if status:
        dirty_files = [line.strip() for line in status.splitlines() if line.strip()]
    return {
        "present": True,
        "commit": commit,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
    }


def _collect_env_files(root: Path) -> list[dict]:
    patterns = [
        "environment.yml",
        "environment*.yml",
        "requirements*.txt",
        "poetry.lock",
        "conda-lock.yml",
        "conda-lock.yaml",
        "Pipfile.lock",
        "*.lock",
        "*conda*.yml",
        "*conda*.yaml",
        "*requirements*.txt",
    ]
    seen: set[Path] = set()
    env_files: list[dict] = []
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_dir():
                continue
            if path in seen:
                continue
            seen.add(path)
            try:
                env_files.append(
                    {
                        "path": _safe_rel(path, root),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
            except Exception:
                env_files.append({"path": _safe_rel(path, root), "error": "unreadable"})
    env_files.sort(key=lambda x: x.get("path", ""))
    return env_files


def _collect_config_files(root: Path) -> list[str]:
    configs: set[Path] = set()
    for pattern in ("config*.yaml", "config*.yml"):
        configs.update(root.rglob(pattern))
    return sorted([_safe_rel(p, root) for p in configs if p.is_file()])


def _parse_snakefile(root: Path) -> dict:
    snakefile = root / "Snakefile"
    if not snakefile.exists():
        return {"present": False}
    text = snakefile.read_text(errors="replace")
    lines = text.splitlines()
    rules: list[dict] = []
    current = None
    for idx, line in enumerate(lines):
        match = re.match(r"^\s*rule\s+([^\s:]+)\s*:", line)
        if match:
            if current:
                rules.append(current)
            current = {"name": match.group(1), "start_line": idx + 1, "shell": [], "script": []}
        if current is not None:
            current.setdefault("lines", []).append(line)
    if current:
        rules.append(current)

    for rule in rules:
        rule_lines = rule.pop("lines", [])
        shell_cmds = []
        script_refs = []
        i = 0
        while i < len(rule_lines):
            line = rule_lines[i]
            shell_match = re.match(r"^\s*shell:\s*(.*)$", line)
            script_match = re.match(r"^\s*(script|notebook|wrapper):\s*(.*)$", line)
            if script_match:
                script_refs.append(script_match.group(2).strip().strip("'\""))
                i += 1
                continue
            if shell_match:
                rest = shell_match.group(1).strip()
                if rest.startswith(("'''", '"""')):
                    quote = rest[:3]
                    buf = []
                    rest = rest[3:]
                    if rest.endswith(quote):
                        buf.append(rest[:-3])
                    else:
                        if rest:
                            buf.append(rest)
                        i += 1
                        while i < len(rule_lines):
                            line2 = rule_lines[i]
                            if quote in line2:
                                before, _ = line2.split(quote, 1)
                                buf.append(before)
                                break
                            buf.append(line2)
                            i += 1
                    shell_cmds.append("\n".join(buf).strip())
                else:
                    shell_cmds.append(rest.strip().strip("'\""))
                i += 1
                continue
            i += 1
        rule["shell"] = [cmd for cmd in shell_cmds if cmd]
        rule["script"] = [ref for ref in script_refs if ref]

    return {"present": True, "path": _safe_rel(snakefile, root), "rules": rules}


def _collect_logs(root: Path, max_files: int, deep: bool) -> list[dict]:
    patterns = ["*.log", "*.out", "*.err"]
    bases = [root / "logs", root]
    if deep:
        bases.extend([root / "outputs", root / "results"])
    logs = []
    seen = set()
    for base in bases:
        if not base.exists():
            continue
        for pattern in patterns:
            for path in sorted(base.rglob(pattern)):
                if len(logs) >= max_files:
                    break
                if not path.is_file() or path in seen:
                    continue
                seen.add(path)
                entry = {"path": _safe_rel(path, root), "bytes": path.stat().st_size}
                try:
                    with path.open("r", errors="replace") as f:
                        head = [next(f).rstrip("\n") for _ in range(8)]
                    entry["head"] = head
                except Exception:
                    entry["head"] = []
                logs.append(entry)
    return logs


def _collect_runtime_hints(root: Path, logs: list[dict], max_hits: int = 200) -> list[dict]:
    patterns = re.compile(r"(elapsed|wall|gpu|cuda|nvidia|device|seconds|minutes|time:)", re.IGNORECASE)
    hits = []
    for entry in logs:
        if len(hits) >= max_hits:
            break
        path = root / entry.get("path", "")
        if not path.exists():
            continue
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            if patterns.search(line):
                hits.append(
                    {
                        "path": entry.get("path"),
                        "line": idx,
                        "text": line.strip()[:200],
                    }
                )
                if len(hits) >= max_hits:
                    break
    return hits


def _collect_best_params(root: Path) -> list[dict]:
    patterns = [
        "**/best_params.json",
        "**/best_config.json",
        "**/neural_ode_best_params.json",
        "**/optuna_trials.json",
    ]
    found = []
    seen = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_dir() or path in seen:
                continue
            seen.add(path)
            entry: dict = {"path": _safe_rel(path, root)}
            try:
                if path.suffix == ".json":
                    data = json.loads(path.read_text(errors="replace"))
                    if isinstance(data, dict):
                        entry["keys"] = sorted(list(data.keys()))
                    elif isinstance(data, list):
                        entry["count"] = len(data)
                entry["bytes"] = path.stat().st_size
            except Exception:
                entry["error"] = "unreadable"
            found.append(entry)
    found.sort(key=lambda x: x.get("path", ""))
    return found


def _collect_sr_sweep_outputs(root: Path, max_files: int, deep: bool) -> list[dict]:
    patterns = [
        "**/best_config.json",
        "**/hall_of_fame*.csv",
        "**/*pareto*.csv",
        "**/*equations*.csv",
        "**/*pysr*.csv",
    ]
    # Everything the pipeline produces now lands under data/. The old root-level
    # scratch directories (results/, outputs/, model/, tuning_runs/) were the default
    # output locations of AI-Feynman, PySR and the Optuna tuner leaking into the working
    # directory; each now writes inside data/ instead, and the historical copies moved to
    # archive/superseded/. They are still scanned when --deep is passed, so an old
    # snapshot can be reproduced, but they are no longer expected to exist.
    roots = [root / "data"]
    if deep:
        roots += [
            root / "wandb",
            root / "archive" / "superseded",
        ]
    found = []
    seen = set()
    for base in roots:
        if not base.exists():
            continue
        for pattern in patterns:
            for path in base.rglob(pattern):
                if len(found) >= max_files:
                    break
                if not path.is_file() or path in seen:
                    continue
                seen.add(path)
                found.append(
                    {
                        "path": _safe_rel(path, root),
                        "bytes": path.stat().st_size,
                    }
                )
    found.sort(key=lambda x: x.get("path", ""))
    return found


def _count_csv(path: Path) -> dict | None:
    try:
        size = path.stat().st_size
    except Exception:
        return None
    if size > 300_000_000:
        return {"path": str(path), "skipped": "too_large", "bytes": size}
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            rows = 0
            for _ in reader:
                rows += 1
        return {"path": str(path), "rows": rows, "cols": len(header), "header": header[:10]}
    except Exception:
        return {"path": str(path), "error": "read_failed"}


def _collect_dataset_sizes(root: Path, max_files: int, deep: bool) -> list[dict]:
    roots = [root / "data"]
    if deep:
        roots.extend([root / "outputs", root / "results"])
    files = []
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*.csv"):
            if path.is_file():
                files.append(path)
    files = sorted(files)[:max_files]
    sizes = []
    for path in files:
        entry = _count_csv(path)
        if entry:
            entry["path"] = _safe_rel(path, root)
            sizes.append(entry)
    return sizes


def _summarize_dataset_sizes(dataset_sizes: list[dict]) -> list[dict]:
    group_keys = [
        "dataset_size_regimes",
        "noise_regimes",
        "mm_deviation_regimes",
        "kinetic_regimes",
        "tqssa",
        "dynamic",
        "static",
        "experimental",
        "three_step_enzyme",
        "panmodel_plots",
        "degradation",
    ]
    buckets: dict[str, dict] = {}
    for entry in dataset_sizes:
        path = entry.get("path")
        rows = entry.get("rows")
        if not path or rows is None:
            continue
        parts = path.split("/")
        key = None
        for g in group_keys:
            if g in parts:
                key = g
                break
        if key is None:
            key = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
        bucket = buckets.setdefault(key, {"group": key, "files": 0, "total_rows": 0, "examples": []})
        bucket["files"] += 1
        bucket["total_rows"] += rows
        if len(bucket["examples"]) < 3:
            bucket["examples"].append(path)
    return sorted(buckets.values(), key=lambda x: x["group"])


def _search_patterns(root: Path, patterns: dict[str, str], max_hits: int, deep: bool) -> dict:
    dirs = [root / "src"]
    if deep:
        dirs.extend([root / "notebooks", root / "scripts", root / "sweeps"])
    results: dict[str, list[dict]] = {k: [] for k in patterns}
    for base in dirs:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix in {".py", ".yaml", ".yml", ".txt", ".md", ".json", ".ipynb"}:
                try:
                    text = path.read_text(errors="replace")
                except Exception:
                    continue
                lines = text.splitlines()
                for idx, line in enumerate(lines, start=1):
                    for key, pat in patterns.items():
                        if len(results[key]) >= max_hits:
                            continue
                        if re.search(pat, line, flags=re.IGNORECASE):
                            results[key].append(
                                {
                                    "path": _safe_rel(path, root),
                                    "line": idx,
                                    "text": line.strip()[:200],
                                }
                            )
    return results


def _collect_metric_defs(root: Path, deep: bool, max_hits: int = 200) -> list[dict]:
    dirs = [root / "src"]
    if deep:
        dirs.extend([root / "notebooks", root / "scripts"])
    hits = []
    regex = re.compile(r"^\s*def\s+([a-zA-Z0-9_]*rmae[a-zA-Z0-9_]*|[a-zA-Z0-9_]*r2[a-zA-Z0-9_]*)\s*\(", re.IGNORECASE)
    for base in dirs:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            try:
                text = path.read_text(errors="replace")
            except Exception:
                continue
            for idx, line in enumerate(text.splitlines(), start=1):
                match = regex.match(line)
                if match:
                    hits.append(
                        {
                            "path": _safe_rel(path, root),
                            "line": idx,
                            "def": match.group(1),
                            "text": line.strip()[:200],
                        }
                    )
                    if len(hits) >= max_hits:
                        return hits
    return hits


def _collect_seed_info(root: Path, deep: bool, max_hits: int = 300) -> dict:
    dirs = [root / "src", root / "sweeps", root / "config.yaml"]
    if deep:
        dirs.extend([root / "notebooks", root / "scripts"])
    seed_lists = []
    seed_values = []
    list_patterns = [
        re.compile(r"\bseed_values\b\s*[:=]\s*\[([^\]]+)\]", re.IGNORECASE),
        re.compile(r"\bseed_list\b\s*[:=]\s*\[([^\]]+)\]", re.IGNORECASE),
        re.compile(r"\bseeds?\b\s*[:=]\s*\[([^\]]+)\]", re.IGNORECASE),
        re.compile(r"--seeds?\s+([0-9\s,]+)", re.IGNORECASE),
    ]
    single_patterns = [
        re.compile(r"np\.random\.seed\((\d+)\)"),
        re.compile(r"random\.seed\((\d+)\)"),
        re.compile(r"torch\.manual_seed\((\d+)\)"),
        re.compile(r"\bseed\s*=\s*(\d+)"),
    ]

    def _extract_ints(chunk: str) -> list[int]:
        return [int(x) for x in re.findall(r"\d+", chunk)]

    for base in dirs:
        if not base:
            continue
        if isinstance(base, Path) and base.is_file():
            files = [base]
        else:
            files = []
            base_path = base if isinstance(base, Path) else Path(base)
            if base_path.exists():
                files.extend(base_path.rglob("*"))
        for path in files:
            if not isinstance(path, Path) or not path.is_file():
                continue
            if path.suffix not in {".py", ".yaml", ".yml", ".json", ".txt", ".ipynb"}:
                continue
            try:
                text = path.read_text(errors="replace")
            except Exception:
                continue
            lines = text.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                for pat in list_patterns:
                    match = pat.search(line)
                    if match:
                        vals = _extract_ints(match.group(1))
                        if vals:
                            seed_lists.append(
                                {
                                    "path": _safe_rel(path, root),
                                    "line": i + 1,
                                    "seeds": vals,
                                    "text": line.strip()[:200],
                                }
                            )
                if re.match(r"^\s*seeds?\s*:\s*$", line):
                    vals = []
                    j = i + 1
                    while j < len(lines) and re.match(r"^\s*-\s*\d+", lines[j]):
                        vals.extend(_extract_ints(lines[j]))
                        j += 1
                    if vals:
                        seed_lists.append(
                            {
                                "path": _safe_rel(path, root),
                                "line": i + 1,
                                "seeds": vals,
                                "text": line.strip()[:200],
                            }
                        )
                for pat in single_patterns:
                    match = pat.search(line)
                    if match:
                        seed_values.append(
                            {
                                "path": _safe_rel(path, root),
                                "line": i + 1,
                                "seed": int(match.group(1)),
                                "text": line.strip()[:200],
                            }
                        )
                i += 1
                if len(seed_lists) >= max_hits and len(seed_values) >= max_hits:
                    break
            if len(seed_lists) >= max_hits and len(seed_values) >= max_hits:
                break
    return {
        "seed_lists": seed_lists[:max_hits],
        "seed_values": seed_values[:max_hits],
    }


def _collect_model_files(root: Path) -> list[dict]:
    base = root / "src" / "synthetic" / "base_models"
    model_files = []
    if base.exists():
        for path in base.rglob("*.py"):
            if not path.is_file():
                continue
            try:
                model_files.append(
                    {
                        "path": _safe_rel(path, root),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
            except Exception:
                model_files.append({"path": _safe_rel(path, root), "error": "unreadable"})
    model_files.sort(key=lambda x: x.get("path", ""))
    return model_files


def _snakemake_fig_rules(snakemake: dict) -> list[dict]:
    rules = snakemake.get("rules", []) if isinstance(snakemake, dict) else []
    fig_rules = []
    for rule in rules:
        name = rule.get("name", "")
        if re.search(r"(fig|figure|table|plot)", name, flags=re.IGNORECASE):
            fig_rules.append(
                {
                    "name": name,
                    "start_line": rule.get("start_line"),
                    "shell": rule.get("shell", []),
                    "script": rule.get("script", []),
                }
            )
    return fig_rules


def _snakemake_dry_runs(root: Path, targets: list[str], config_file: str | None, max_lines: int = 200) -> list[dict]:
    runs = []
    for target in targets:
        cmd = ["snakemake", "-n", "--printshellcmds", target]
        if config_file:
            cmd.extend(["--configfile", config_file])
        result = _run_cmd_full(cmd, root)
        stdout = (result.get("stdout") or "").splitlines()
        stderr = (result.get("stderr") or "").splitlines()
        runs.append(
            {
                "target": target,
                "command": result.get("command"),
                "returncode": result.get("returncode"),
                "error": result.get("error"),
                "stdout_head": stdout[:max_lines],
                "stderr_head": stderr[:max_lines],
                "stdout_tail": stdout[-max_lines:] if len(stdout) > max_lines else [],
                "stderr_tail": stderr[-max_lines:] if len(stderr) > max_lines else [],
            }
        )
    return runs


def _summarize(snapshot: dict) -> str:
    lines: list[str] = []
    lines.append(f"Generated: {snapshot.get('generated_at')}")
    lines.append(f"Repo: {snapshot.get('repo_root')}")
    lines.append("")
    git = snapshot.get("git", {})
    lines.append("Git")
    lines.append(f"- present: {git.get('present')}")
    if git.get("commit"):
        lines.append(f"- commit: {git.get('commit')}")
    lines.append(f"- dirty: {git.get('dirty')}")
    if git.get("dirty_files"):
        lines.append(f"- dirty_files: {len(git.get('dirty_files'))}")
    lines.append("")
    versions = snapshot.get("versions", {})
    lines.append("Versions")
    lines.append(f"- python: {versions.get('python')}")
    if versions.get("julia"):
        lines.append(f"- julia: {versions.get('julia')}")
    if versions.get("snakemake"):
        lines.append(f"- snakemake: {versions.get('snakemake')}")
    packages = versions.get("packages", {})
    if isinstance(packages, dict):
        for key in sorted(packages):
            lines.append(f"- {key}: {packages.get(key)}")
    missing = versions.get("missing_packages", [])
    if missing:
        lines.append(f"- missing (not installed & not in env files): {', '.join(sorted(missing))}")
    lines.append("")
    lines.append("Env Files")
    for item in snapshot.get("env_files", [])[:20]:
        lines.append(f"- {item.get('path')} ({item.get('bytes')} bytes)")
    if len(snapshot.get("env_files", [])) > 20:
        lines.append(f"- ... ({len(snapshot.get('env_files', [])) - 20} more)")
    lines.append("")
    snk = snapshot.get("snakemake", {})
    lines.append("Snakemake")
    lines.append(f"- Snakefile: {snk.get('path') if snk.get('present') else 'missing'}")
    if snk.get("present"):
        rules = snk.get("rules", [])
        lines.append(f"- rules: {len(rules)}")
        fig_rules = snapshot.get("snakemake_fig_rules", [])
        lines.append(f"- fig/table/plot rules: {len(fig_rules)}")
    lines.append("")
    lines.append("Best Params / Sweeps")
    for item in snapshot.get("best_params_files", [])[:20]:
        lines.append(f"- {item.get('path')}")
    if len(snapshot.get("best_params_files", [])) > 20:
        lines.append(f"- ... ({len(snapshot.get('best_params_files', [])) - 20} more)")
    lines.append("")
    lines.append("Dataset Sizes (sample)")
    for item in snapshot.get("dataset_sizes", [])[:10]:
        if "rows" in item:
            lines.append(f"- {item.get('path')}: {item.get('rows')} rows, {item.get('cols')} cols")
        else:
            lines.append(f"- {item.get('path')}: {item.get('skipped') or item.get('error')}")
    if len(snapshot.get("dataset_sizes", [])) > 10:
        lines.append(f"- ... ({len(snapshot.get('dataset_sizes', [])) - 10} more)")
    lines.append("")
    lines.append("Pattern Hits (sample)")
    for key, hits in snapshot.get("pattern_hits", {}).items():
        lines.append(f"- {key}: {len(hits)} hits")
    lines.append("")
    seed_info = snapshot.get("seed_info", {})
    if seed_info:
        lines.append("Seeds")
        lines.append(f"- seed_lists: {len(seed_info.get('seed_lists', []))}")
        lines.append(f"- seed_values: {len(seed_info.get('seed_values', []))}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect reproducibility snapshot.")
    parser.add_argument("--root", default=None, help="Repo root (default: parent of this script)")
    parser.add_argument("--out", default="repro_snapshot.json", help="JSON output path")
    parser.add_argument("--text", default="repro_snapshot.txt", help="Text summary output path")
    parser.add_argument("--max-files", type=int, default=200, help="Max files scanned per category")
    parser.add_argument("--deep", action="store_true", help="Deeper scan into outputs/results/notebooks")
    parser.add_argument(
        "--snakemake-targets",
        default="",
        help="Comma-separated snakemake targets to dry-run (requires --snakemake-dry-run).",
    )
    parser.add_argument(
        "--snakemake-dry-run",
        action="store_true",
        help="Run snakemake -n --printshellcmds for provided targets.",
    )
    parser.add_argument(
        "--snakemake-config",
        default="config.yaml",
        help="Config file for snakemake dry-run (default: config.yaml if present).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]

    env_files = _collect_env_files(root)
    env_versions = _collect_versions_from_env_files(root, env_files)
    versions = _collect_versions(root)
    packages_installed = versions.get("packages", {})
    missing = []
    for key, val in packages_installed.items():
        if val is None and not env_versions.get(key):
            missing.append(key)
    versions["packages_from_env_files"] = env_versions
    versions["missing_packages"] = missing

    snakemake = _parse_snakefile(root)
    fig_rules = _snakemake_fig_rules(snakemake)

    targets = [t.strip() for t in args.snakemake_targets.split(",") if t.strip()]
    config_path = args.snakemake_config if args.snakemake_config and (root / args.snakemake_config).exists() else None
    snakemake_runs = []
    if args.snakemake_dry_run and targets:
        snakemake_runs = _snakemake_dry_runs(root, targets, config_path)

    logs = _collect_logs(root, args.max_files, args.deep)
    runtime_hints = _collect_runtime_hints(root, logs)
    dataset_sizes = _collect_dataset_sizes(root, args.max_files, args.deep)

    snapshot: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "git": _collect_git(root),
        "versions": versions,
        "env_files": env_files,
        "config_files": _collect_config_files(root),
        "snakemake": snakemake,
        "snakemake_fig_rules": fig_rules,
        "snakemake_dry_runs": snakemake_runs,
        "logs": logs,
        "runtime_hints": runtime_hints,
        "best_params_files": _collect_best_params(root),
        "sr_sweep_outputs": _collect_sr_sweep_outputs(root, args.max_files, args.deep),
        "model_file_hashes": _collect_model_files(root),
        "seed_info": _collect_seed_info(root, args.deep),
        "dataset_sizes": dataset_sizes,
        "dataset_size_summary": _summarize_dataset_sizes(dataset_sizes),
        "metric_definitions": _collect_metric_defs(root, args.deep),
        "pattern_hits": _search_patterns(
            root,
            {
                "rmae": r"\brmae\b",
                "r2": r"\br2\b",
                "seed": r"\bseed\b",
                "snakemake": r"\bsnakemake\b",
                "figure_table": r"\b(fig|figure|table)\b",
            },
            max_hits=200,
            deep=args.deep,
        ),
        "limits": {
            "max_files": args.max_files,
            "deep": args.deep,
            "notes": [
                "CSV row counts limited to files <= 300MB.",
                "Pattern hits capped per pattern.",
                "Snakefile parsing is best-effort (simple regex).",
                "Snakemake dry-run only runs if --snakemake-dry-run is set.",
                "Env file parsing is best-effort (regex-based).",
            ],
        },
    }

    out_path = root / args.out
    out_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))

    text_path = root / args.text
    text_path.write_text(_summarize(snapshot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
