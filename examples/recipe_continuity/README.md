# Grouping OCR text into recipes

A worked example of strands decider on a document task. OCR on a scanned page gives you text
boxes, a title here and a paragraph there, but not the recipes they belong to. Layout tells you
where each box is. Strands decider reads the text and says whether a recipe continues from one
box to the next.

```bash
git clone https://github.com/strands-labs/strands-decider && cd strands-decider
pip install -e .
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099

python examples/recipe_continuity/recipe_continuity.py
```

The script uses only the Python standard library. It expects a server on port 8099 unless
`STRANDS_DECIDER_URL` is set, and refuses any address that is not `http://` or `https://`.

The page, `arabic_cookbook.json`, is a synthetic page from an invented Arabic family cookbook,
read right to left. It has 16 boxes: a page header (labelled `null`, no recipe) and 9
hand-labelled recipes and tips. It was written for this example and includes a few traps:

- a recipe whose steps continue at the top of the next column (to the left, since it's Arabic)
- a paragraph that the OCR split in the middle of a sentence
- a second egg dish directly under the end of the first
- three short kitchen tips stacked one under another, each its own item

## How it works

1. **Layout proposes.** For each box, the candidate is the box directly below it in the same
   column, or, if nothing is below, the top of the next column. The full-width page header
   belongs to no recipe and is left out.
2. **The decider judges.** It sees the end of one box and the start of the next, plus one line
   of layout context, and answers with a probability.
3. **Code settles the links.** Claims are accepted strongest first, and each box can be continued
   by only one other box. The links chain into recipes.

## `noul` to decide, `choice` to rank

A `noul` scores its own two criteria, `true` and `false`, so its probability is an absolute
judgment of the one question "does B continue A?". A `choice` spreads one distribution over the
options you supply, so each value is a share of the total, and `none` is scored like any other
option string. With one candidate plus `none`, the model weighs a concrete text against an
abstract option; with several candidates, read the top value against 1/N.

So the script uses `noul` to decide whether a link exists, and `choice` only to rank several
candidates. `--question` forces one shape everywhere, to compare them:

```bash
python examples/recipe_continuity/recipe_continuity.py                    # 9/9 recipes
python examples/recipe_continuity/recipe_continuity.py --question choice  # 6/9 recipes
```

With `strands-decider-2B-qwen3.5-v1-2610`, the `choice` run misses the recipe title and the
column change, and links two separate recipes. The criteria wording matters too. The `false`
criterion names what starts a new item on this page: a dish name followed by a full stop or a
dash, or the word نصيحة (tip). With the generic "B starts a different recipe, a new title, or a
separate tip." the default run rebuilds 5 of 9. Background is in
[issue #132](https://github.com/strands-labs/strands-decider/issues/132).

> This example is an illustration rather than a benchmark: one synthetic page, and questions and
> thresholds picked by hand. Every accepted link scores between 0.5 and 0.9, the band the
> [routing convention](../../docs/architecture.md#the-routing-convention) says to confirm before
> acting, so treat the links as proposals.
