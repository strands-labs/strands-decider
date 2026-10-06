"""cli.main: a finished `ask` leaves through os._exit; everything else exits normally."""

from __future__ import annotations

import subprocess
import sys

import pytest

from strands_decider import cli


class _Exited(Exception):
    def __init__(self, code):
        self.code = code


@pytest.fixture
def fake_exit(monkeypatch):
    """Records the atexit run and the os._exit, in order; neither really happens here."""
    events = []

    def _exit(code):
        events.append("os._exit")
        raise _Exited(code)

    monkeypatch.setattr(cli.os, "_exit", _exit)
    monkeypatch.setattr(cli.atexit, "_run_exitfuncs", lambda: events.append("atexit"))
    monkeypatch.setattr(cli, "_exit_fast", False)
    return events


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
    assert fake_exit == ["atexit", "os._exit"]


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


def test_the_fast_exit_leaks_no_semaphore():
    # Under spawn (the macOS default) tqdm's multiprocessing lock is a named semaphore
    # registered with the resource tracker. os._exit without the atexit handlers leaves it
    # registered, and the tracker warns once the process is gone.
    script = (
        "import multiprocessing as mp; mp.set_start_method('spawn')\n"
        "from tqdm import tqdm; tqdm.get_lock()\n"
        "from strands_decider import cli\n"
        "def app():\n"
        "    cli._exit_fast = True\n"
        "cli.app = app\n"
        "cli.main()\n"
    )
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=120)
    assert res.returncode == 0, res.stderr
    assert "leaked semaphore" not in res.stderr


def _tiny_checkpoint(tmp_path):
    """A random-weight Qwen3 decider checkpoint whose base model is a local directory."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

    from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel

    vocab = {w: i for i, w in enumerate(["<pad>", "<eos>", "<unk>", "a", "b", "c"])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>",
                                  unk_token="<unk>")
    base = tmp_path / "base"
    Qwen3Model(Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                           num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                           head_dim=8)).save_pretrained(base)
    cfg = StrandsDeciderConfig(base_model=str(base), head_type="pointer", pointer_dim=8,
                               torch_dtype="float32", lora_r=2)
    model = StrandsDeciderModel(cfg, Qwen3Model.from_pretrained(base), tok)
    model.attach_lora()
    model.save_pretrained(str(tmp_path / "ckpt"))
    return str(tmp_path / "ckpt")


def test_a_real_ask_takes_the_fast_exit(fake_exit, monkeypatch, tmp_path, capsys):
    # With a fake `app`, ask_cmd could stop setting _exit_fast and the tests above would
    # still pass. This one runs the console-script entry point on a real checkpoint.
    ckpt = _tiny_checkpoint(tmp_path)
    monkeypatch.setattr(sys, "argv", ["strands-decider", "ask", ckpt, "--state", "a b c",
                                      "--noul", "a b?", "--device", "cpu"])
    with pytest.raises(_Exited) as e:
        cli.main()
    assert e.value.code == 0 and cli._exit_fast
    assert fake_exit == ["atexit", "os._exit"]
    assert "noul_0 noul =" in capsys.readouterr().out
