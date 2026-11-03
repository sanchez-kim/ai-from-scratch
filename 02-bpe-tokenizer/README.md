# 02 · BPE tokenizer and what Korean costs

A byte-level BPE tokenizer in about 70 lines of plain Python
(`bpe/tokenizer.py`: train merges, encode, decode), three tests, and token
counts for English prose, Korean prose and Python code with four tokenizers:

| Name | What it is |
|---|---|
| `ours-en` | our BPE, 8,000 merges trained on 542 KB of Simple English Wikipedia |
| `ours-mixed` | our BPE, 8,000 merges trained on 541 KB: half that English text, half Korean Wikipedia |
| `cl100k_base` | OpenAI's encoding, via `tiktoken` |
| `o200k_base` | OpenAI's newer encoding, via `tiktoken` |

Everything measured here is a count. No timing is recorded, and a second run
gives the same numbers.

## Run it

```bash
uv venv && uv sync
uv run pytest
uv run python download_data.py        # ~3 MB into data/ (gitignored), plus a 25 MB archive read in memory
uv run python measure_tokens.py       # writes results/token-counts.json and results/merges-*.txt
uv run python check_roundtrip.py      # decode(encode(x)) == x on every file; writes results/roundtrip.json
uv run python figures/make_merge_animation.py
uv run python figures/make_counts_chart.py
```

`tiktoken` downloads the `cl100k_base` and `o200k_base` vocabulary files on
first use.

## Data and licences

`download_data.py` fetches three sources into `data/`. None of the text is
committed. `results/corpus-manifest.json` records every Wikipedia page's
revision ID and the sha256 of each saved file, so you can tell whether a
later download got the same text.

| File | Source | Licence |
|---|---|---|
| `wiki-en-{train,test}.txt` | [Simple English Wikipedia](https://simple.wikipedia.org), 163 topics, plain-text extracts, headings removed, first 6,000 characters per article | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `wiki-ko-{train,test}.txt` | [Korean Wikipedia](https://ko.wikipedia.org), the Korean article linked from each of the same 163 topics | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `python-code.txt` | 10 modules from [CPython v3.12.7 `Lib/`](https://github.com/python/cpython/tree/v3.12.7/Lib) | [PSF License 2.0](https://github.com/python/cpython/blob/v3.12.7/LICENSE) |
| `flores-{eng,kor}.txt` | [FLORES-200](https://github.com/facebookresearch/flores/tree/main/flores200) devtest, 1,012 sentences, English and Korean only | [CC BY-SA 4.0](https://github.com/facebookresearch/flores#licenses) |

Every 4th topic (in list order) is held out for testing. The tokenizers never
see the test articles, the code or FLORES.

## Results

- `results/hypothesis.md`: expectations, written before the first count.
- `results/token-counts.json`: tokens per 1,000 characters, the FLORES
  comparison, the vocabulary-size sweep, the 한국어 example, vocabulary facts,
  and the `uptime` line before and after the run.
- `results/merges-en.txt`, `results/merges-mixed.txt`: the learned merges, one
  per line as `id_a id_b new_id`, in learning order.
- `results/roundtrip.json`: round-trip check on every data file.
- `results/pytest.log`: the test run below.

## Test log

```
$ uv run pytest -v
platform darwin -- Python 3.12.13, pytest-9.2.0, pluggy-1.7.0
tests/test_tokenizer.py::test_known_merge_sequence_and_an_unseen_word PASSED [ 33%]
tests/test_tokenizer.py::test_round_trip_for_any_utf8_text PASSED        [ 66%]
tests/test_tokenizer.py::test_english_merges_never_join_hangul_bytes PASSED [100%]
============================== 3 passed in 0.01s ===============================
```

## License

Code: MIT, see [../LICENSE](../LICENSE). The downloaded data keeps its own
licence (table above).
