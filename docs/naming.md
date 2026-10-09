# Model names

Released models are named

```
strands-decider-{size}-{base}-v{N}-{YYMM}
```

for example `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`.

| part | meaning | values so far |
| --- | --- | --- |
| `size` | the base model's own size token, as its Hub id writes it | `2B` (Qwen/Qwen3.5-2B-Base); `E2B`, `E4B`, `12B`, `26B-A4B` (google/gemma-4-*) |
| `base` | the base model family, one token | `qwen3.5`, `gemma4` |
| `v{N}` | the public recipe generation | `v1` |
| `YYMM` | the release month, as on the Hub (Qwen's `-2507`) | `2610` |

- **`v{N}`** counts recipes, not training runs. It goes up only when a change is visible to
  users: new kinds of training data (code, for example), a new head, a new objective. The
  same recipe has the same N on every base and size, so `strands-decider-E4B-gemma4-v1-2610`
  and `strands-decider-2B-qwen3.5-v1-2610` are the same recipe on two bases.
- **A second release in the same month** goes to the same repository. Every upload is
  tagged on the Hub with its date, `YYMMDD`, and the card gets a changelog line. To pin a
  version, download it by its tag ([inference.md](inference.md#model-artifact)).
- **Internal round numbers** stay in the research notes. They are not part
  of the public name.
- **Earlier releases keep their names:** `strands-decider-2B-hobson-v19` and
  `strands-decider-2B-hobson-v21`. hobson-v21 and v1 are different recipes; v1 is not a
  rename of v21.

`strands_decider.naming.release_name(base_model, N, yymm)` builds a name from the base
model's Hub id, and `strands_decider.naming.parse(name)` checks one:

```bash
python -m strands_decider.naming Qwen/Qwen3.5-2B-Base 1 2610   # strands-decider-2B-qwen3.5-v1-2610
```

Git tags for model releases are `model/<name>`, for example
`model/strands-decider-2B-qwen3.5-v1-2610`. Package versions use `v<number>` tags only, so
a model tag never sets the package version.
