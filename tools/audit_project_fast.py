#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path


EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".venv",
    "venv",
    "env",
    "audit_md",

    "dataset",
    "datasets",
    "dataset_augmented",
    "dataset_preprocessed",
    "dataset_preprocessed_v2",
    "datasets_gan_mix",
    "datasets_gan_mix_raw",

    "gan_data",
    "gan_data_raw",
    "gan_data_raw_zips",
    "gan_analysis",
    "gan_training_runs",
    "gan_training_runs_batch32",
    "gan_training_runs_quality",
    "gan_training_runs_transfer",

    "diploma_figures",
    "diploma_results",
    "diploma_results__full",

    "gradcam_results",
    "gradcam_diploma",
    "binary_threshold_analysis_20_raw_supcon",
    "threshold_analysis_20_raw_supcon",
    "synthetic_mixing_runs",

    "stylegan3-brecahad",
}

EXCLUDED_FILES = {
    "presentation_figures.tar.gz",
}

SOURCE_EXTS = {".py", ".sh", ".bash", ".zsh"}
CONFIG_EXTS = {
    ".yml", ".yaml", ".json", ".toml", ".ini", ".cfg", ".conf", ".md", ".txt"
}
CSV_EXTS = {".csv"}
NOTEBOOK_EXTS = {".ipynb"}

PROJECT_FILENAMES = {
    "Dockerfile",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
    "Makefile",
    "makefile",
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "poetry.lock",
    "Pipfile",
    "Pipfile.lock",
    "setup.py",
    "setup.cfg",
    "environment.yml",
    "environment.yaml",
    ".gitignore",
    ".dockerignore",
    "README.md",
    "readme.md",
}

SENSITIVE_FILES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "credentials.json",
    "secrets.json",
    "service-account.json",
}

SECRET_RE = re.compile(
    r"(?i)^(\s*[^#\n]*?"
    r"(password|passwd|pwd|token|secret|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|client[_-]?secret|authorization|bearer)"
    r"[^=\n:]*\s*[:=]\s*)(.+)$",
    re.MULTILINE,
)


def run_cmd(cmd: list[str], cwd: Path, timeout: int = 10) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", f"Timeout: {' '.join(cmd)}"
    except FileNotFoundError:
        return 127, "", f"Command not found: {cmd[0]}"


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def human_size_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"


def file_size(path: Path) -> str:
    try:
        return human_size_bytes(path.stat().st_size)
    except OSError:
        return "unknown"


def is_excluded(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False

    if path.name in EXCLUDED_FILES:
        return True

    return any(part in EXCLUDED_DIRS for part in parts)


def is_sensitive(path: Path) -> bool:
    name = path.name
    return name in SENSITIVE_FILES or (name.startswith(".env") and name != ".env.example")


def mask_secrets(text: str) -> str:
    return SECRET_RE.sub(r"\1***MASKED***", text)


def looks_binary(path: Path) -> bool:
    try:
        sample = path.read_bytes()[:4096]
    except OSError:
        return True
    return b"\x00" in sample


def read_text(path: Path, max_bytes: int) -> tuple[str | None, str | None]:
    if is_sensitive(path):
        return None, "sensitive file skipped"

    try:
        size = path.stat().st_size
    except OSError as e:
        return None, f"cannot stat: {e}"

    if size > max_bytes:
        return None, f"too large: {file_size(path)}"

    if looks_binary(path):
        return None, "binary file skipped"

    for enc in ("utf-8", "cp1251"):
        try:
            return mask_secrets(path.read_text(encoding=enc)), None
        except UnicodeDecodeError:
            pass
        except OSError as e:
            return None, f"cannot read: {e}"

    return None, "cannot decode as utf-8/cp1251"


def fence(text: str) -> str:
    longest = 2
    for m in re.finditer(r"`+", text):
        longest = max(longest, len(m.group(0)))
    return "`" * (longest + 1)


def code_block(text: str, lang: str = "") -> str:
    f = fence(text)
    return f"{f}{lang}\n{text.rstrip()}\n{f}\n"


def lang_for(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()

    if suffix == ".py":
        return "python"
    if suffix in {".sh", ".bash", ".zsh"}:
        return "bash"
    if suffix in {".yml", ".yaml"}:
        return "yaml"
    if suffix == ".json":
        return "json"
    if suffix == ".toml":
        return "toml"
    if suffix in {".ini", ".cfg", ".conf"}:
        return "ini"
    if suffix == ".md":
        return "markdown"
    if name == "dockerfile":
        return "dockerfile"
    if name == "makefile":
        return "makefile"
    return ""


def iter_files(root: Path):
    for current_root, dirs, files in os.walk(root):
        current = Path(current_root)

        dirs[:] = [
            d for d in dirs
            if not is_excluded(current / d, root)
        ]

        for name in files:
            path = current / name
            if not is_excluded(path, root):
                yield path


def classify(root: Path) -> dict[str, list[Path]]:
    result = {
        "source": [],
        "config": [],
        "csv": [],
        "notebooks": [],
        "other": [],
    }

    for p in iter_files(root):
        suffix = p.suffix.lower()

        if suffix in SOURCE_EXTS:
            result["source"].append(p)
        elif suffix in CONFIG_EXTS or p.name in PROJECT_FILENAMES:
            result["config"].append(p)
        elif suffix in CSV_EXTS:
            result["csv"].append(p)
        elif suffix in NOTEBOOK_EXTS:
            result["notebooks"].append(p)
        else:
            result["other"].append(p)

    for key in result:
        result[key].sort(key=lambda x: rel(x, root))

    return result


def limit_lines(text: str, max_lines: int = 500) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n... truncated, total lines: {len(lines)}"


def report_index(root: Path, out: Path, classified: dict[str, list[Path]]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""# Индекс быстрого аудита проекта

Дата: `{now}`

Корень проекта: `{root}`

Папка отчета: `{out}`

## Найдено

- Source py/sh/bash/zsh: `{len(classified["source"])}`
- Конфиги и проектные файлы: `{len(classified["config"])}`
- CSV: `{len(classified["csv"])}`
- Jupyter notebooks: `{len(classified["notebooks"])}`
- Прочие файлы: `{len(classified["other"])}`

## Файлы отчета

- `01_TREE.md`
- `02_GIT_STATUS.md`
- `03_SOURCE_FILES.md`
- `04_CONFIG_FILES.md`
- `05_CSV_OVERVIEW.md`
- `06_NOTEBOOKS.md`
- `07_EXCLUDED.md`
- `08_NESTED_GIT_REPOS.md`
- `09_OTHER_FILES.md`

## Важно

Это быстрый аудит. Он не раскрывает тяжелые директории и не считает их размер рекурсивно.
"""


def report_tree(root: Path, max_depth: int) -> str:
    lines = ["# Дерево проекта\n\n"]

    def walk(path: Path, prefix: str, depth: int):
        if depth > max_depth:
            lines.append(f"{prefix}└── ... depth limit\n")
            return

        try:
            entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except OSError as e:
            lines.append(f"{prefix}└── [cannot read: {e}]\n")
            return

        for i, e in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            child_prefix = "    " if i == len(entries) - 1 else "│   "
            excluded = is_excluded(e, root)

            if e.is_dir():
                marker = " [excluded]" if excluded else ""
                lines.append(f"{prefix}{connector}{e.name}/{marker}\n")
                if not excluded:
                    walk(e, prefix + child_prefix, depth + 1)
            else:
                marker = " [excluded]" if excluded else ""
                lines.append(f"{prefix}{connector}{e.name} ({file_size(e)}){marker}\n")

    walk(root, "", 1)
    return "".join(lines)


def report_git(root: Path) -> str:
    lines = ["# Git-состояние\n\n"]

    code, out, err = run_cmd(["git", "rev-parse", "--is-inside-work-tree"], root)
    if code != 0 or out != "true":
        lines.append("Папка не является Git-репозиторием.\n")
        if err:
            lines.append(code_block(err))
        return "".join(lines)

    commands = [
        ("Текущая ветка", ["git", "branch", "--show-current"]),
        ("Последний commit", ["git", "log", "-1", "--oneline"]),
        ("Remote", ["git", "remote", "-v"]),

        # Важно: без -uall, иначе Git может долго обходить dataset и результаты.
        ("Краткий статус без untracked", ["git", "status", "--short", "-uno"]),

        ("Staged", ["git", "diff", "--cached", "--name-status"]),
        ("Modified tracked", ["git", "diff", "--name-status"]),

        # Tracked обычно быстрый и важный.
        ("Tracked files", ["git", "ls-files"]),

        # --directory показывает не каждый файл внутри больших папок, а директории.
        ("Untracked compact", ["git", "ls-files", "--others", "--exclude-standard", "--directory"]),
        ("Ignored compact", ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "--directory"]),
    ]

    for title, cmd in commands:
        lines.append(f"\n## {title}\n\n")
        code, out, err = run_cmd(cmd, root, timeout=15)

        if code == 0:
            lines.append(code_block(limit_lines(out if out else "Пусто.")))
        else:
            lines.append(f"Команда завершилась с кодом `{code}`.\n\n")
            lines.append(code_block(err if err else "No stderr"))

    return "".join(lines)


def report_text_files(title: str, files: list[Path], root: Path, max_bytes: int) -> str:
    lines = [f"# {title}\n\n"]

    if not files:
        lines.append("Файлы не найдены.\n")
        return "".join(lines)

    for p in files:
        lines.append(f"\n## `{rel(p, root)}`\n\n")
        lines.append(f"Размер: `{file_size(p)}`\n\n")

        text, reason = read_text(p, max_bytes)
        if reason:
            lines.append(f"_Содержимое пропущено: {reason}._\n")
        else:
            lines.append(code_block(text or "", lang_for(p)))

    return "".join(lines)


def report_csv(files: list[Path], root: Path, max_rows: int = 10) -> str:
    lines = ["# CSV overview\n\n"]

    if not files:
        lines.append("CSV-файлы не найдены.\n")
        return "".join(lines)

    for p in files:
        lines.append(f"\n## `{rel(p, root)}`\n\n")
        lines.append(f"Размер: `{file_size(p)}`\n\n")

        try:
            with p.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                rows = []
                for i, row in enumerate(reader):
                    rows.append(row)
                    if i >= max_rows:
                        break
        except UnicodeDecodeError:
            try:
                with p.open("r", encoding="cp1251", newline="") as f:
                    reader = csv.reader(f)
                    rows = []
                    for i, row in enumerate(reader):
                        rows.append(row)
                        if i >= max_rows:
                            break
            except Exception as e:
                lines.append(f"_Не удалось прочитать CSV: {e}._\n")
                continue
        except Exception as e:
            lines.append(f"_Не удалось прочитать CSV: {e}._\n")
            continue

        if not rows:
            lines.append("_CSV пустой._\n")
            continue

        csv_text = "\n".join(",".join(row) for row in rows)
        lines.append(code_block(mask_secrets(csv_text), "csv"))

    return "".join(lines)


def report_notebooks(files: list[Path], root: Path, max_bytes: int) -> str:
    lines = ["# Jupyter notebooks\n\n"]

    if not files:
        lines.append("Ноутбуки не найдены.\n")
        return "".join(lines)

    for p in files:
        lines.append(f"\n## `{rel(p, root)}`\n\n")
        lines.append(f"Размер: `{file_size(p)}`\n\n")

        try:
            size = p.stat().st_size
        except OSError:
            lines.append("_Не удалось определить размер._\n")
            continue

        if size == 0:
            lines.append("_Пустой notebook._\n")
            continue

        if size > max_bytes:
            lines.append("_Notebook слишком большой, содержимое пропущено._\n")
            continue

        try:
            nb = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            lines.append(f"_Не удалось прочитать notebook: {e}._\n")
            continue

        cells = nb.get("cells", [])
        lines.append(f"Количество ячеек: `{len(cells)}`\n\n")

        for i, cell in enumerate(cells, start=1):
            cell_type = cell.get("cell_type", "unknown")
            source = "".join(cell.get("source", []))

            if not source.strip():
                continue

            lines.append(f"### Cell {i}: {cell_type}\n\n")

            if cell_type == "code":
                lines.append(code_block(mask_secrets(source), "python"))
            else:
                lines.append(mask_secrets(source).strip() + "\n\n")

    return "".join(lines)


def report_excluded(root: Path) -> str:
    lines = ["# Исключенные элементы\n\n"]
    lines.append("Эти элементы не раскрывались и не обходились рекурсивно.\n\n")

    for name in sorted(EXCLUDED_DIRS):
        p = root / name
        if p.exists():
            lines.append(f"- `{name}/` — excluded\n")

    for name in sorted(EXCLUDED_FILES):
        p = root / name
        if p.exists():
            lines.append(f"- `{name}` — excluded, size `{file_size(p)}`\n")

    return "".join(lines)


def report_nested_git(root: Path) -> str:
    lines = ["# Вложенные Git-репозитории\n\n"]

    found = []

    # Только прямые подпапки корня, без глубокого обхода.
    try:
        for child in root.iterdir():
            if child.is_dir() and (child / ".git").exists():
                found.append(child)
    except OSError as e:
        return f"# Вложенные Git-репозитории\n\nНе удалось прочитать корень: {e}\n"

    if not found:
        lines.append("Вложенные Git-репозитории в прямых подпапках не найдены.\n")
        return "".join(lines)

    for repo in sorted(found, key=lambda p: p.name.lower()):
        lines.append(f"\n## `{rel(repo, root)}`\n\n")

        for title, cmd in [
            ("Remote", ["git", "remote", "-v"]),
            ("Branch", ["git", "branch", "--show-current"]),
            ("Last commit", ["git", "log", "-1", "--oneline"]),
            ("Status", ["git", "status", "--short", "-uno"]),
        ]:
            lines.append(f"### {title}\n\n")
            code, out, err = run_cmd(cmd, repo, timeout=10)
            if code == 0:
                lines.append(code_block(out if out else "Пусто."))
            else:
                lines.append(code_block(err if err else f"Exit code {code}"))

    return "".join(lines)


def report_other(files: list[Path], root: Path) -> str:
    lines = ["# Прочие файлы\n\n"]

    if not files:
        lines.append("Прочие файлы не найдены.\n")
        return "".join(lines)

    for p in files:
        lines.append(f"- `{rel(p, root)}` — {file_size(p)}\n")

    return "".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="audit_md")
    parser.add_argument("--max-file-kb", type=int, default=250)
    parser.add_argument("--tree-depth", type=int, default=3)
    return parser.parse_args()


def write(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    max_bytes = args.max_file_kb * 1024

    classified = classify(root)

    reports = {
        "00_INDEX.md": report_index(root, out, classified),
        "01_TREE.md": report_tree(root, args.tree_depth),
        "02_GIT_STATUS.md": report_git(root),
        "03_SOURCE_FILES.md": report_text_files(
            "Source-файлы py/sh/bash/zsh",
            classified["source"],
            root,
            max_bytes,
        ),
        "04_CONFIG_FILES.md": report_text_files(
            "Конфиги и проектные файлы",
            classified["config"],
            root,
            max_bytes,
        ),
        "05_CSV_OVERVIEW.md": report_csv(classified["csv"], root),
        "06_NOTEBOOKS.md": report_notebooks(classified["notebooks"], root, max_bytes),
        "07_EXCLUDED.md": report_excluded(root),
        "08_NESTED_GIT_REPOS.md": report_nested_git(root),
        "09_OTHER_FILES.md": report_other(classified["other"], root),
    }

    for filename, text in reports.items():
        write(out / filename, text)

    print(f"Готово. Отчет сохранен в: {out}")
    print(f"Главный файл: {out / '00_INDEX.md'}")


if __name__ == "__main__":
    main()