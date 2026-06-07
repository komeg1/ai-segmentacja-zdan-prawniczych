# Description
Use this skill to tune hyper-parameters in `spacy_config/config.cfg` and evaluate model performance.

# Context
The current model is a `senter` pipeline using `spacy.HashEmbedCNN.v2`. We are using `pl_core_news_lg` vectors. 

# Instructions
1. When asked to evaluate, expect to read `data/spacy_ready/dev.spacy` (which we will create from the gold standard).
2. If the model overfits on the Silver Data, suggest exact modifications to `config.cfg` (e.g., dropout, batch size, compounding schedules).
3. Do not rewrite Python code for training. Always suggest using the CLI command: `python -m spacy train spacy_config/config.cfg --output ./models/ --paths.train ./data/spacy_ready/train.spacy --paths.dev ./data/spacy_ready/dev.spacy`.