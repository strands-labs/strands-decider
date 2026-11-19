"""The v2 recipe's pieces: token-readout optimizer defaults."""

from __future__ import annotations

import yaml

import training.train as T


def _cfg(tmp_path, **raw):
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(raw))
    return T.TrainConfig.from_yaml(str(tmp_path / "c.yaml"))


def _opt(cfg):
    return (cfg.head_dropout, cfg.head_lr, cfg.weight_decay, tuple(cfg.adam_betas))


def test_a_token_readout_trains_with_the_adapter_by_default(tmp_path):
    cfg = _cfg(tmp_path, head_type="token_readout", lr=3e-5)
    T._readout_defaults(cfg)
    assert _opt(cfg) == (0.0, 3e-5, 0.0, (0.9, 0.999))


def test_set_keys_win_and_the_pointer_defaults_are_unchanged(tmp_path):
    cfg = _cfg(tmp_path, head_type="token_readout", head_lr=1e-3, adam_betas=[0.9, 0.95])
    T._readout_defaults(cfg)
    assert _opt(cfg) == (0.0, 1e-3, 0.0, (0.9, 0.95))
    pointer = _cfg(tmp_path, head_type="pointer")
    T._readout_defaults(pointer)
    assert _opt(pointer) == (0.05, 1e-3, 0.01, (0.9, 0.95))
