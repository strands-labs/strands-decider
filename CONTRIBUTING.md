# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Security issue notifications
If you discover a potential security issue in this project, follow [SECURITY.md](SECURITY.md). Please do **not** create a public GitHub issue.


## Reporting Bugs/Feature Requests

We welcome you to file a [Bug Report](../../issues/new?template=bug_report.yml) or a [Feature Request](../../issues/new?template=feature_request.yml).

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used (commit id if you are on `main`)
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment


## Using AI Tools

We build with coding agents too, and you're welcome to use them. But **you are the author of your pull request, not your agent.** Before you open a PR, make sure you understand the code well enough to explain why it works, defend the design choices, and maintain it if asked. If you couldn't walk a reviewer through it line by line, it's not ready yet.

A few things that help:

- **Keep changes small and incremental.** A focused PR that does one thing is far easier to review than a large one that touches many areas. When in doubt, split it up.
- **Open an issue first for anything significant**, so we can align on the approach before you (or your agent) invest the time. For a change to the training recipe, the data, or the model, that means a preregistration ([Experiments](#experiments)).
- **Review every line your agent generates.** Delete what you don't need, simplify what's over-engineered, and make sure tests actually exercise the behaviour — not just pass. Trim comments that narrate the agent's reasoning: a comment should state only what a reader cannot infer from the code.

The bug, feature, and PR templates each ask for a short "Human Overview" written in your own words when an agent drafted the body. That is where you confirm that you understood what you are submitting.


## Contributing via Pull Requests
Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the `main` branch, the default branch. Pull requests target `main`.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source; please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Ensure local tests pass ([Development](#development)).
4. Commit to your fork using clear commit messages.
5. Send us a pull request, answering any default questions in the pull request interface.
6. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.

GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).


## Development

The package needs Python 3.11 or later. Install it with the `dev` extra, which adds pytest, ruff, pre-commit and commitizen. Training and the data build need the `train` extra as well ([training/README.md](training/README.md#setup)):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install -t pre-commit -t commit-msg   # once per clone; see below
pytest -q                  # GPU tests skip automatically without CUDA
pytest -q -m distributed   # multi-process training tests on CPU, several minutes
ruff check .               # the whole tree, as CI does
mypy ./src                 # the published package only, as CI does
```

On an Apple-silicon Mac, `pip install -e ".[dev,mlx]"` also runs `tests/test_mlx_engine.py`, which skips elsewhere, CI included. On Linux without an NVIDIA GPU, install the CPU build of torch first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`. The suite downloads nothing from Hugging Face. CI runs `pytest -q` with `HF_HUB_OFFLINE=1` on Python 3.11 and 3.12; it does not run the distributed tests, so run those yourself. A separate CI job runs `ruff check .` over the whole tree and `mypy ./src`, with the latest ruff that `pyproject.toml` admits; both must pass.

No test checks the documents. If you move a file or rename a heading, search the markdown files and the code comments for the old path or heading and fix every reference.

Keep mechanical changes (moves, renames, formatting) and functional changes in separate commits, with an imperative summary and a body that explains why. Do not change the model inputs, the calibration or the training behaviour in a pull request that says it does something else.

### hatch (optional)

The wheel is built with setuptools; you do not need [hatch](https://hatch.pypa.io/) to develop against the package. But if you install it (`pipx install hatch`), the shortcuts in `pyproject.toml` cover the common loops:

```bash
hatch run test               # pytest -q
hatch run test-distributed   # pytest -q -m distributed
hatch run lint:check         # ruff check . and mypy ./src, as CI runs them
hatch run format             # ruff format
```

### pre-commit and commit messages

Local hooks live in [`.pre-commit-config.yaml`](.pre-commit-config.yaml) and run on the files you commit, not the whole tree. Install them once:

```bash
pre-commit install -t pre-commit -t commit-msg
```

The `pre-commit` stage runs `ruff --fix` and `ruff format` on staged files. The `commit-msg` stage runs commitizen and enforces [Conventional Commits](https://www.conventionalcommits.org/): a subject like `feat: add score threshold` or `fix(inference): handle empty state`. The allowed types are `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, and `revert`, matching the PR-title gate in [`.github/workflows/pr-title.yml`](./.github/workflows/pr-title.yml). If a commit message is rejected, `git commit --amend` to fix it — the hook does not otherwise gate history.

## Experiments

The reference model is v19, and its recipe is `configs/train.yaml` with `training/recipe.sh`. A change to the recipe, the data or the model is an experiment, and experiments here are preregistered ([research/README.md](research/README.md)): the predictions and the failure conditions are committed before training, and the outcome is appended after the run without editing what came before. By that rule, a run that misses its bar does not replace the reference model; the maintainers decide any exception, and the record says so.

To propose one, open an issue first. State the change, the predictions, and what would refute them, in the form of the files in `research/preregistrations/`. Once agreed, commit the preregistration file, with its configuration under `configs/experiments/`, before you train. Training the recipe needs an NVIDIA GPU on Linux or WSL2 for about 11 hours ([training/README.md](training/README.md)), so agree on the plan before you spend the compute. Rank the result on JevBench, not on the repository's own held-out sets ([evaluation/README.md](evaluation/README.md)).

## Finding contributions to work on
Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any 'help wanted' issues is a great place to start.


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Licensing

The code is licensed under the Apache License, Version 2.0; see [LICENSE](LICENSE). By submitting a contribution, you agree to license it under the same terms. The data sources and their licences are listed in [data/sources.md](data/sources.md).
