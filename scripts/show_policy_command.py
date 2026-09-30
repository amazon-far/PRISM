#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

try:
    import tkinter as tk
except Exception as exc:
    print(f"[command_window] tkinter unavailable: {exc}", file=sys.stderr)
    raise SystemExit(0)


def _read_status(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError):
        return {}


def main() -> None:
    status_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/holosoma_policy_command_status.json")

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"[command_window] cannot open display: {exc}", file=sys.stderr)
        return

    root.title("Policy sparse root command")
    root.geometry("420x210+30+30")
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass

    title = tk.Label(root, text="Policy command", font=("TkDefaultFont", 16, "bold"))
    title.pack(anchor="w", padx=14, pady=(12, 2))

    command_label = tk.Label(root, text="waiting...", font=("TkFixedFont", 24, "bold"), justify="left")
    command_label.pack(anchor="w", padx=14, pady=(4, 8))

    detail_label = tk.Label(root, text=str(status_path), font=("TkDefaultFont", 10), justify="left", wraplength=390)
    detail_label.pack(anchor="w", padx=14, pady=(0, 12))

    def update() -> None:
        status = _read_status(status_path)
        command = status.get("command")
        timestamp = float(status.get("timestamp", 0.0) or 0.0)
        age = time.time() - timestamp if timestamp else None

        if isinstance(command, list) and len(command) >= 3:
            command_label.config(
                text=(
                    f"x   {float(command[0]):+0.3f}\n"
                    f"y   {float(command[1]):+0.3f}\n"
                    f"yaw {float(command[2]):+0.3f}"
                )
            )
            stale = age is not None and age > 0.5
            mode = "stale" if stale else "live"
            detail_label.config(
                text=(
                    f"{mode} | {status.get('term', 'unknown')}\n"
                    f"manual={status.get('manual_offset', [])} joystick={status.get('joystick_offset', [])}\n"
                    f"force_zero={status.get('force_zero_sparse_root_command', False)}"
                )
            )
        else:
            command_label.config(text="waiting...")
            detail_label.config(text=str(status_path))

        root.after(50, update)

    update()
    root.mainloop()


if __name__ == "__main__":
    main()
