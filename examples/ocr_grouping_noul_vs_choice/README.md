# Grouping OCR text: `noul` to decide, `choice` to rank

OCR gives you text boxes, not the recipes they belong to. This example groups the boxes of a
synthetic Arabic cookbook page (right to left, three columns) in three steps:

1. **Layout proposes** a next box for each box: the box below it, or the top of the next column.
2. **Strands decider judges** each candidate pair with a `noul`: P(box B continues box A's recipe).
3. **Code settles** the links, strongest first, at most one next and one previous box per box.

```bash
pip install -e .
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099

# in a second terminal, once the model has loaded
python examples/ocr_grouping_noul_vs_choice/group_ocr_text.py                    # 9/9 recipes
python examples/ocr_grouping_noul_vs_choice/group_ocr_text.py --question choice  # 6/9 recipes
```

The script uses only the Python standard library, and talks to the server at
`STRANDS_DECIDER_URL` (default `http://127.0.0.1:8099`).

## `noul` to decide, `choice` to rank

A `noul` is a softmax over its two criteria, `true` and `false`, for the one pair it is asked
about. A `choice` is one softmax over all the options you give it, so a `none` option is scored
like any other option string. By default the script uses a `noul` to decide whether a link
exists. When a box has several candidates, it first ranks them with a `choice` (without `none`),
then asks the `noul` about the top one.

On this page every box has one candidate. `--question choice` asks "this block, or `none`?"
instead, and with `strands-decider-2B-qwen3.5-v1-2610` it misses the link from a title to its
text and the jump to the next column, and links two separate recipes: 6 of 9 instead of 9.

The criteria wording matters as much. The `false` criterion in `SAME` names what starts a new
item on this page (a dish name followed by a full stop or dash, or the word نصيحة, "tip"). With
a generic "B starts a different recipe, a new title, or a separate tip." the default run gets 5
of 9. Background is in [issue #132](https://github.com/strands-labs/strands-decider/issues/132).

## The page

`arabic_cookbook.json` has 16 boxes, each with an `id`, a `bbox` (`[x0, y0, x1, y1]`, pixels),
its `text`, and a hand label `recipe` (`null` for the page header) used only for scoring. Its
traps: a recipe that continues in the next column, a paragraph split mid-sentence, a second dish
right under the first, and three short tips stacked as separate items.

> This is an illustration, not a benchmark: one synthetic page, with criteria tuned to it. The
> accepted links score between 0.54 and 0.86, so treat them as proposals to check (see the
> [routing convention](../../docs/architecture.md#the-routing-convention)).
