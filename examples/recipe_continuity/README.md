# Grouping OCR text into recipes

**In short:** use a `noul` to decide whether one link exists ("does box B continue box A?"),
and a `choice` only to pick between several real candidates. This example shows why, on a
document task.

OCR on a scanned page gives you text boxes, a title here and a paragraph there, but not the
recipes they belong to. Layout tells you where each box is. Strands decider reads the text and
says whether a recipe continues from one box to the next.

```bash
git clone https://github.com/strands-labs/strands-decider && cd strands-decider
pip install -e .
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099

# in a second terminal, from the repository root
python examples/recipe_continuity/recipe_continuity.py
```

The script uses only the Python standard library. It expects a server on port 8099 unless
`STRANDS_DECIDER_URL` is set, and refuses any address that is not `http://` or `https://`.

## The page

`arabic_cookbook.json` is a synthetic page from an invented Arabic family cookbook, read right
to left in three columns. It has 16 boxes: a page header (labelled `null`, no recipe) and 9
hand-labelled recipes and tips. It was written for this example and includes a few traps:

- a recipe whose steps continue at the top of the next column (to the left, since it's Arabic)
- a paragraph that the OCR split in the middle of a sentence
- a second egg dish directly under the end of the first
- three short kitchen tips stacked one under another, each its own item

Each box has an `id`, a `bbox` of `[x0, y0, x1, y1]` in pixels, its `text`, and the
hand label `recipe`, which the script uses only to score its output. The ids are not in
reading order. `--page` takes another file in the same format; set `direction` to `"ltr"`
for a left-to-right page, and tune the layout settings at the top of the script to its scale.

## How it works

1. **Layout proposes.** For each box, the candidate is the box directly below it in the same
   column, or, if nothing is below, the top of the next column. The page header spans the
   page, belongs to no recipe, and is left out.
2. **The decider judges.** It sees the end of one box and the start of the next, plus one line
   of layout context, and answers with a probability.
3. **Code settles the links.** Links are accepted strongest first. Each box gets at most one
   next box and at most one previous box, and the links chain into recipes.

A run prints each group of box ids and whether it matches a labelled recipe:

```text
ok = matches a labelled recipe, xx = does not, -- = not part of any recipe

ok k01 k02 k03 k04  شكشوكة
ok k05 k06 k07      عجة الأعشاب
ok k13              نصيحة: يُحفظ الخبز في كيس
...
-- k00              كتاب المطبخ العائلي — الصفحة

9/9 recipes match the labels (auto questions)
```

## `noul` to decide, `choice` to rank

A `noul` is a softmax over its two criteria, `true` and `false`, so its probability is the
share of `true` for the one pair it was asked about. A `choice` is one softmax over all the
options you supply, so each value is a share of the total, and `none` is scored like any other
option string.

So the script asks a `noul` when a box has one candidate, and a `choice` when it has several,
keeping the top option unless that is `none`. On this page every box has at most one
candidate, so the default run asks only `noul` questions. `--question choice` forces a `choice`
everywhere, which here means one candidate against `none`:

```bash
python examples/recipe_continuity/recipe_continuity.py                    # 9/9 recipes
python examples/recipe_continuity/recipe_continuity.py --question choice  # 6/9 recipes
```

With `strands-decider-2B-qwen3.5-v1-2610`, the `choice` run misses the link from the recipe
title to its text and the jump to the next column, and links two separate recipes. In our
reading, the model is weighing a concrete text against an abstract `none`. The criteria wording
matters too. The `false` criterion names what starts a new item on this page: a dish name
followed by a full stop or a dash, or the word نصيحة (tip). With the generic "B starts a
different recipe, a new title, or a separate tip." the default run rebuilds 5 of 9. Background
is in [issue #132](https://github.com/strands-labs/strands-decider/issues/132).

> This example is an illustration rather than a benchmark: one synthetic page, with questions,
> criteria and thresholds tuned by hand to it. Every accepted link scores between 0.54 and
> 0.86, and the closest rejected link scores 0.47, so treat the links as proposals to check,
> not as decisions (see the [routing convention](../../docs/architecture.md#the-routing-convention)).
