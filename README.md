#### Note: This is a WIP repository with unfunctional code at the moment.

This repository contains scripts to convert [pre-trained translation models](https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models) that are trained using [Marian NMT](https://github.com/marian-nmt) to pytorch format.

An example execution command is provided below:

`python convert_marian_to_pytorch.py --model_dir . --weights_file_name opus.spm32k-spm32k.transformer-align.model1.npz.best-perplexity.npz --vocab_file_name opus.spm32k-spm32k.vocab.yml --source_tokenizer_model source.spm --target_tokenizer_model target.spm --text_to_translate "hello world"`
