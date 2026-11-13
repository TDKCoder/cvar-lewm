#!/usr/bin/env python3
"""Replace host-specific details with placeholders before anything is committed (decision 2026-11-13).

    python3 scripts/redact.py FILE...            # redact in place
    python3 scripts/redact.py --copy SRC DST     # copy a run output into results/ redacted
    python3 scripts/redact.py --check            # scan tracked + staged files; exit 1 on a hit

Placeholders: <user>, <uid>, <home>, <node>. Generic patterns (cluster node / file-server hostnames,
NFS export and /dsmlp home paths, host=... fields) live here. The literal username and UID live in
the git-ignored docs/local/redact.txt (one `literal<TAB>placeholder` per line, # comments), so this
public script never contains them. Public service hostnames (dsmlp-docs, dsmlp-login, reserve,
datahub) are left alone.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "docs" / "local" / "redact.txt"
END = r"[^\s`)'\",;]*"  # rest of a path token

GENERIC = [  # order matters: whole NFS export specs before bare hostnames
    (re.compile(r"its-dsmlp-fs\d+(?:\.ucsd\.edu)?:/export/" + END), "<node>:<home>"),
    (re.compile(r"/dsmlp/(?:home|workspaces)-fs\d+/" + END), "<home>"),
    (re.compile(r"its-dsmlp-[a-z]+\d+(?:\.ucsd\.edu)?"), "<node>"),
    (re.compile(r"\b(host(?:name)?=)(?!<node>|\$)\S+"), r"\1<node>"),  # not code like host=$(hostname)
]


def local_rules() -> list[tuple[re.Pattern, str]]:
    if not LOCAL.is_file():
        sys.exit(f"missing {LOCAL} (git-ignored: literal<TAB>placeholder per line)")
    rules = []
    for line in LOCAL.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            lit, ph = line.split("\t")
            if ph == "<user>":  # /home/<literal>[/...] is a home directory
                rules.append((re.compile(r"/home/" + re.escape(lit) + r"(?![\w.-])"), "<home>"))
            rules.append((re.compile(r"(?<![\w.-])" + re.escape(lit) + r"(?![\w.-])"), ph))
    return rules


def redact(text: str, rules) -> str:
    for rx, ph in GENERIC + rules:
        text = rx.sub(ph, text)
    return text


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--cached"], capture_output=True, check=True)
    return [ROOT / p for p in out.stdout.decode().split("\0") if p]


def main(argv: list[str]) -> int:
    rules = local_rules()
    if argv[:1] == ["--check"]:
        hits = 0
        for f in tracked_files():
            if not f.is_file() or f.suffix in {".png", ".pdf", ".npz", ".pt", ".ckpt"}:
                continue
            try:
                lines = f.read_text().splitlines()
            except UnicodeDecodeError:
                continue
            for i, line in enumerate(lines, 1):
                if redact(line, rules) != line:
                    hits += 1
                    print(f"{f.relative_to(ROOT)}:{i}: host-specific detail (run scripts/redact.py on it)")
        print(f"redact --check: {hits} hit(s)" if hits else "redact --check: clean")
        return 1 if hits else 0
    if argv[:1] == ["--copy"] and len(argv) == 3:
        src, dst = Path(argv[1]), Path(argv[2])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(redact(src.read_text(), rules))
        return 0
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    for name in argv:
        p = Path(name)
        old = p.read_text()
        new = redact(old, rules)
        if new != old:
            p.write_text(new)
            print(f"redacted {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
