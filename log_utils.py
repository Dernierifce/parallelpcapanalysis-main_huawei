"""Shared helpers for tee-style file logging."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO


DEFAULT_LOG_NAME = "log.txt"


@dataclass
class Tee:
    streams: tuple[TextIO, ...]

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def setup_run_logging(outdir: str | Path, script_name: str, log_file: str | None = None) -> Path:
    """Mirror stdout/stderr to a log file stored in the output directory."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_file) if log_file else output_dir / DEFAULT_LOG_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_handle = open(log_path, "a", encoding="utf-8")
    header = f"\n[{datetime.now().isoformat(timespec='seconds')}] {script_name}\n"
    log_handle.write(header)
    log_handle.flush()

    sys.stdout = Tee((sys.stdout, log_handle))
    sys.stderr = Tee((sys.stderr, log_handle))
    return log_path


def emit_report(title: str, sections: dict[str, object], width: int = 70) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

    for section_name, content in sections.items():
        print(f"\n[{section_name}]")
        if isinstance(content, dict):
            for key, value in content.items():
                print(f"  - {key}: {value}")
        elif isinstance(content, (list, tuple)):
            for item in content:
                print(f"  - {item}")
        elif content is not None:
            print(f"  {content}")

    print("=" * width)