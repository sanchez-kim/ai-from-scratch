"""Header for every results file: when, where, and how busy the machine was."""

import datetime
import platform
import subprocess


def uptime():
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def header(script):
    return {
        "script": script,
        "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "uptime_before": uptime(),
    }
