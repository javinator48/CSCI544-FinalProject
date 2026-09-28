# Multilingual Named Entity Recognition

Final project for **CSCI 544 (Applied Natural Language Processing)**. We tag people, organizations and locations in sentences from five languages: English, French, Spanish, Japanese and Chinese. We compare several neural sequence-labeling architectures built on top of multilingual BERT, and combine them with weak-supervision aggregation. We also compare against prompting ChatGPT to do the tagging directly.

## Overview

- **Task:** token-level NER with BIO tags: `PER`, `ORG`, `LOC` (and `MISC`, which doesn't occur in our data).
- **Data:** [WikiANN](https://huggingface.co/datasets/wikiann) splits for five languages, merged into one multilingual train, dev and test set (`data/merge/`).
- **Encoder:** `bert-base-multilingual-cased` (mBERT), fine-tuned on the merged data. Subword embeddings are pooled back to one vector per word.
- **Downstream taggers** trained on top of the mBERT embeddings:
  - BiLSTM
  - BiLSTM + character-level CNN
  - BiLSTM + CRF
  - Character CNN + Transformer encoder (self-attention)
  - Attention + CRF, and Attention (Transformer encoder) + CRF
  - Dilated CNN (DCNN)
  - Multi-channel CNN
- **Ensembling:** each model's predictions are treated as a noisy labeler and aggregated with [skweak](https://github.com/NorskRegnesentral/skweak)'s hidden Markov model.
- **LLM baseline:** `gpt-3.5-turbo`, prompted to act as a multilingual NER tagger.

## Results

All scores are entity-level, computed with [seqeval](https://github.com/chakki-works/seqeval) (`evaluate_experiments.ipynb`).

| Model | Dev F1 | Test precision | Test recall | Test F1 |
|---|---|---|---|---|
| BiLSTM | 0.774 | 0.739 | 0.774 | 0.756 |
| BiLSTM + CharCNN | 0.787 | 0.754 | 0.783 | 0.768 |
| CharCNN + Attention (Encoder) | 0.748 | 0.702 | 0.762 | 0.730 |
| **BiLSTM + CRF** | **0.837** | **0.834** | 0.808 | **0.821** |
| Attention + CRF | 0.780 | 0.787 | 0.742 | 0.763 |
| Attention (Encoder) + CRF | 0.804 | 0.805 | 0.769 | 0.786 |
| Multi-channel CNN | 0.708 | 0.660 | 0.718 | 0.687 |
| Dilated CNN | 0.685 | 0.618 | 0.732 | 0.670 |
| skweak HMM ensemble (BiLSTM, BiLSTM + CharCNN, BiLSTM + CRF, CharCNN + Attention) | 0.836 | — | — | 0.816 |

**Findings:**
- **Adding a CRF layer helped most.** It lifted the BiLSTM from 0.756 to 0.821 test F1, and the attention encoder from 0.730 to 0.786, because the CRF learns which tag sequences are valid.
- **Character-level features helped modestly.** BiLSTM + CharCNN beat the plain BiLSTM by about 1 point.
- **The HMM ensemble matched but didn't beat the best single model.** It scored 0.836 dev F1, the same as BiLSTM + CRF.
- **`ORG` was the hardest entity type** for every model.

### ChatGPT baseline

We prompted `gpt-3.5-turbo` (`GPTExperiment/promptGPT.py`) on small samples of the training data (`GPTExperiment/chatgptResults/`):

| Language | Sentences | Precision | Recall | F1 |
|---|---|---|---|---|
| English | 88 | 0.415 | 0.500 | 0.453 |
| Spanish | 94 | 0.391 | 0.535 | 0.452 |
| Japanese | 61 | 0.293 | 0.433 | 0.349 |
| French | 94 | 0.295 | 0.411 | 0.343 |

Zero-shot prompting did far worse than the supervised models. These numbers come from small samples, so they aren't directly comparable to the full test-set results above.

## Repository layout

| Path | Contents |
|---|---|
| `data/<language>/` | Raw WikiANN parquet splits per language |
| `data/merge/` | Merged multilingual train, dev, test and raw splits, plus tag and character vocabularies |
| `data/mBERT/` | mBERT tokenizer and config files. The `.bin` weights are git-ignored. |
| `models/mBERT_fine_tuning*.ipynb` | Fine-tuning mBERT on the merged data, including a LoRA variant |
| `downstream_pipeline*.ipynb` | Training the downstream taggers on pooled mBERT word embeddings, one notebook per architecture |
| `models/`, `hhw-utils/` | Model definitions and training scripts (BiLSTM-CRF, CharCNN, Transformer encoder, DCNN, …) |
| `model_prediction_files/` | Each model's predictions on train, dev and test |
| `evaluate_experiments.ipynb` | seqeval evaluation of all prediction files, which produced the table above |
| `skweak/` | HMM aggregation of model predictions, and Hugging Face NER baselines |
| `GPTExperiment/` | The ChatGPT prompting script, outputs and evaluation |
| `finetunebaseline.ipynb` | Reference: mBERT fine-tuned end to end on WikiANN with the Hugging Face `Trainer` |

## Running the code

Install the dependencies. A CUDA GPU is strongly recommended.

```bash
pip install torch transformers datasets evaluate seqeval pandas pyarrow scikit-learn \
            lightning pytorch-crf skweak spacy openai tqdm
```

1. **Fine-tune mBERT.** Run `models/mBERT_fine_tuning.ipynb`, and save the fine-tuned weights to `data/mBERT/fine/`. Model weights (`*.bin`, `*.pt`) aren't committed.
2. **Train a downstream tagger.** Run one of the `downstream_pipeline_diff_seqLen_pooling_*.ipynb` notebooks. Each one writes its predictions to `model_prediction_files/`.
3. **Evaluate.** Run `evaluate_experiments.ipynb`.
4. **Ensemble (optional).** Run `skweak/skweakAggregate.ipynb`.
5. **ChatGPT baseline (optional).** Provide an OpenAI API key through the `OPENAI_API_KEY` environment variable rather than hard-coding it, then run `GPTExperiment/promptGPT.py`.

Some notebooks still contain absolute paths from the machine they were written on. Point them at your local `data/` folder before running.

## Team

Javin Liu, SARIHUST, X2Y, hjzccc, and Aaaaaaamber, as named in the commit history.
