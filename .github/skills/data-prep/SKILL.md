# Description
Use this skill to debug, refactor, and run the NLTK-to-spaCy pipeline to generate Silver Standard data (`train.spacy`).

# Context
We use NLTK (with custom Polish legal abbreviations) to roughly tokenize texts from `data/raw/` into sentences. We then project these boundaries onto spaCy `Doc` objects and save them to a `DocBin`.

# Instructions
1. Analyze `src/build_silver_data.py`.
2. Check for alignment issues where `span = doc.char_span(...)` returns `None`. 
3. Ensure NLTK handles Polish legal enumerations correctly (e.g., "1)", "a)", "art.").
4. If asked to modify the script, ensure the output always goes to `data/spacy_ready/train.spacy`.
5. Point out logical errors bluntly. Provide corrected code blocks.