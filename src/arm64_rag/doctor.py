import os
import platform
import re
import subprocess


def _read(*command: str) -> str | None:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _integer(value: str | None) -> int | None:
    try:
        number = int(value) if value is not None else None
    except ValueError:
        return None
    return number if number is not None and number >= 0 else None


def _version(value: str | None) -> str | None:
    match = re.search(r"\b\d+\.\d+(?:\.\d+)?\b", value or "")
    return match.group() if match else None


def collect_inventory() -> dict:
    system = platform.system()
    hardware = {"logical_cores": os.cpu_count()}
    if system == "Darwin":
        hardware.update(
            cpu_model=_read("sysctl", "-n", "machdep.cpu.brand_string"),
            memory_bytes=_integer(_read("sysctl", "-n", "hw.memsize")),
            performance_cores=_integer(
                _read("sysctl", "-n", "hw.perflevel0.physicalcpu")
            ),
            efficiency_cores=_integer(
                _read("sysctl", "-n", "hw.perflevel1.physicalcpu")
            ),
            features={
                name: _integer(_read("sysctl", "-n", f"hw.optional.arm.FEAT_{flag}"))
                for name, flag in {
                    "dotprod": "DotProd",
                    "i8mm": "I8MM",
                    "sme": "SME",
                }.items()
            },
        )
    return {
        "schema_version": 1,
        "platform": {
            "system": system,
            "version": platform.mac_ver()[0]
            if system == "Darwin"
            else platform.release(),
            "architecture": platform.machine(),
        },
        "hardware": hardware,
        "tools": {
            "python": platform.python_version(),
            **{
                tool: _version(_read(tool, "--version"))
                for tool in ("git", "uv", "cmake", "ninja", "clang")
            },
        },
    }
