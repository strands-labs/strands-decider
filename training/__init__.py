"""Training code for strands-decider.

Lives outside the published `strands_decider` package (which ships inference only).
Run as modules: `python -m training.train` (the trainer, for the token readout and the pointer
head), `python -m training.convert`, `python -m training.merge`, `python -m training.average`,
`python -m training.export`, `python -m training.evaluate`, `python -m training.soup`,
`python -m training.data`.
"""
