"""cli.main: a finished `ask` leaves through os._exit; everything else exits normally."""

from __future__ import annotations

import pytest

from strands_decider import cli


class _Exited(Exception):
    def __init__(self, code):
        self.code = code


@pytest.fixture
def fake_exit(monkeypatch):
    def _exit(code):
        raise _Exited(code)

    monkeypatch.setattr(cli.os, "_exit", _exit)
    monkeypatch.setattr(cli, "_exit_fast", False)


def _app_that(*, finish_ask: bool, code: int | None):
    def app():
        if finish_ask:
            cli._exit_fast = True
        raise SystemExit(code)

    return app


@pytest.mark.parametrize("code", [0, None])
def test_finished_ask_exits_fast(fake_exit, monkeypatch, code):
    monkeypatch.setattr(cli, "app", _app_that(finish_ask=True, code=code))
    with pytest.raises(_Exited) as e:
        cli.main()
    assert e.value.code == 0


@pytest.mark.parametrize("code", [0, 2])
def test_other_commands_exit_normally(fake_exit, monkeypatch, code):
    monkeypatch.setattr(cli, "app", _app_that(finish_ask=False, code=code))
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == code


def test_failure_after_ask_finished_keeps_its_code(fake_exit, monkeypatch):
    # The flag alone is not enough: a nonzero exit still goes through sys.exit.
    monkeypatch.setattr(cli, "app", _app_that(finish_ask=True, code=1))
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 1
