# Hypotheses, written before any token count on the evaluation sets
Written: 2026-11-02 21:28:04 KST

Setup known at this point: corpora downloaded (results/corpus-manifest.json).
Training text: English-only = wiki-en-train (542 KB). Mixed = first half of that
by bytes + the first equal number of bytes of wiki-ko-train. Up to 8,000 merges each.
I have only run training for 300 merges on English as a speed check, and the
three unit tests. No evaluation counts yet.

## H1 English prose (wiki-en-test), tokens per 1,000 characters
- cl100k_base and o200k_base: about 220-250 (the "about 4 characters per token" rule).
- Our English-only BPE at 8,000 merges: worse, about 260-300 (much smaller vocabulary).
- Our mixed BPE: 10-20% worse than English-only on English, because half of its
  merges budget goes to Korean.

## H2 Korean prose (wiki-ko-test)
- The text is about 2.3 UTF-8 bytes per character (2/3 of characters are Hangul).
- Our English-only BPE: close to the byte count, about 2,000-2,300 tokens per
  1,000 characters, at every vocabulary size, because no Hangul byte pair exists
  in its training text. Only spaces, digits and Latin words merge.
- Our mixed BPE: about 600-900.
- cl100k_base: about 900-1,100 (roughly one token per Hangul syllable or worse).
- o200k_base: about 500-700, clearly better than cl100k.

## H3 Python code (CPython stdlib), tokens per 1,000 characters
- cl100k and o200k: about 250-320.
- Both of our BPEs: worse, about 400-550, since neither saw code.

## H4 FLORES-200 devtest, same sentences in English and Korean
- Korean/English total-token ratio: cl100k about 2.0-2.5x, o200k about 1.2-1.5x,
  our English-only about 4-6x, our mixed about 1.3-1.8x.

## H5 Vocabulary sweep (0 to 8,000 merges)
- At 0 merges tokens = bytes. Gains shrink as merges grow (diminishing returns).
- English-only curve on Korean stays almost flat.
