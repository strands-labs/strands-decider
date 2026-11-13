# Test fixtures

- `previous_trainer.json`: inputs and outputs of the trainer the released token-readout models were trained with, for
  the pieces this trainer reimplements, under these keys: the row lengths (`row_length`, after the function), the
  microbatches (`pack_microbatches`, likewise), the step plan and its placement on the GPUs (`plans`: sha256 of the
  plan and of each rank's work), the target matrix (`target_matrix`), the option shuffling (`shuffle_options`), the
  learning-rate schedule (`schedule`), the loss (`loss`, float32 bit patterns), and the dev metrics and the
  temperature calibration (`metrics`, with `metrics.calibrate_temperature`). The tests check that `training/` gives
  the same outputs on the same inputs, bit for bit when `STRANDS_DECIDER_EXACT_FIXTURES=1` under the versions in
  `training/requirements.txt` (`trainer_support.py` says what the tests allow elsewhere). The previous trainer could
  also place a step's microbatches on the GPUs by their estimated cost; this one does not, so the tests check the
  placement of the cases that did not, and the plan of every case.
- `render.json`: 208 edge cases (JSON states, special-token strings, empty and null instructions, letter-named and
  255 options, one-option choices, non-string score levels, NaN and Infinity values, a lone surrogate) with what `strands_decider.core` renders for each under
  both templates: the ids' sha256, their count, the state length S, the code count, or the refusal. The core was at
  `24382ce155648da732e840e6e0e3a0e155b9340f` when the fixture was made. Each row also says whether the core the
  released models were trained with (`50e986452427a6396eaa8871184cc74b4807b553`) rendered it the same; all do except
  the rows whose state holds NaN or Infinity, which the newer core refuses and that core rendered, and the lone
  surrogate, which both refuse with different errors. The rendering test needs Gemma 4's `tokenizer.json` (set
  `STRANDS_DECIDER_GEMMA4_TOKENIZER`); its sha256 is `cc8d3a0c...` for every size.
