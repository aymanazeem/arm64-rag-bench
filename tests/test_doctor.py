import json
import subprocess

import pytest

from arm64_rag import cli, doctor


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(),
        PermissionError(),
        subprocess.TimeoutExpired("sysctl", 5),
        subprocess.CalledProcessError(1, "sysctl"),
    ],
)
def test_unavailable_commands_return_unknown(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(doctor.subprocess, "run", fail)
    assert doctor._read("sysctl", "-n", "hw.memsize") is None


def test_mac_inventory_preserves_unsupported_and_unknown_features(monkeypatch):
    answers = {
        "machdep.cpu.brand_string": "Apple M1 Max",
        "hw.memsize": "34359738368",
        "hw.perflevel0.physicalcpu": "8",
        "hw.perflevel1.physicalcpu": "2",
        "hw.optional.arm.FEAT_DotProd": "1",
        "hw.optional.arm.FEAT_I8MM": "0",
        "hw.optional.arm.FEAT_SME": None,
    }
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(doctor.platform, "mac_ver", lambda: ("26.7.1", (), "arm64"))
    monkeypatch.setattr(doctor, "_read", lambda *args: answers.get(args[-1]))

    result = doctor.collect_inventory()

    assert result["hardware"]["memory_bytes"] == 32 * 1024**3
    assert result["hardware"]["performance_cores"] == 8
    assert result["hardware"]["features"] == {"dotprod": 1, "i8mm": 0, "sme": None}
    assert result["tools"]["clang"] is None


def test_non_mac_inventory_omits_mac_queries_and_tool_paths(monkeypatch):
    commands = []

    def read(*args):
        commands.append(args)
        return "tool version 3.4.5\nInstalled in /Users/private-name/tools"

    monkeypatch.setattr(doctor.platform, "system", lambda: "Linux")
    monkeypatch.setattr(doctor, "_read", read)
    result = doctor.collect_inventory()

    assert all(command[0] != "sysctl" for command in commands)
    assert result["tools"]["cmake"] == "3.4.5"
    assert "private-name" not in json.dumps(result)
    assert "cpu_model" not in result["hardware"]


@pytest.mark.parametrize("value", [None, "", "not available", "-1", "1.5"])
def test_invalid_hardware_numbers_are_unknown(value):
    assert doctor._integer(value) is None


def test_doctor_cli_returns_json(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["arm64-rag", "doctor"])
    monkeypatch.setattr(cli, "collect_inventory", lambda: {"schema_version": 1})
    cli.main()
    assert json.loads(capsys.readouterr().out) == {"schema_version": 1}
