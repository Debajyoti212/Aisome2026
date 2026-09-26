# Data Documentation

## 1. Overview

The Squad4 system uses two categories of data:

1. English labelled stance data.
2. Parallel English-Hindi and English-Bengali data for training the NMT system.

The complete datasets are not included in this repository.

Users should obtain the relevant datasets from their original sources and comply with the corresponding licenses and usage conditions.

---

## 2. English Labelled Stance Data

The classifier training data were assembled from several English stance-data sources.

After removing duplicate text and retaining the three labels used by the AISoMe task, the final English dataset contained:

* 3,008 comments
* 1,106 Favour
* 978 Against
* 924 None

The SemEval-2016 Task 6 route filtered for climate-related targets but returned zero rows in the final data-assembly process.

---

## 3. Translation Data

The NMT system was trained on approximately:

* 980,000 English-Hindi sentence pairs
* 980,000 English-Bengali sentence pairs

The parallel data were derived from Samanantar.

The original experiments used a Kaggle-hosted prepared dataset containing separate training and validation files.

---

## 4. Translation Pre-processing

The NMT pipeline used:

* a shared SentencePiece vocabulary
* 48,000 vocabulary pieces
* character coverage of 0.9995
* NMT-NFKC normalization
* dummy prefix spaces
* `<hi>` as the Hindi target-language tag
* `<bn>` as the Bengali target-language tag

Rows with missing values were removed.

Training sequences were truncated to a maximum of 128 subword tokens including special symbols.

---

## 5. Translated Classifier Data

Every English labelled comment was translated into:

* Hindi
* Bengali

This produced:

* 3,008 Hindi comments
* 3,008 Bengali comments

The two translated datasets were pooled into:

* 6,016 classifier examples

The pooled labels were:

* 2,212 Favour
* 1,956 Against
* 1,848 None

---

## 6. Official Test Data

The official test sets contained:

* 500 Hindi comments
* 500 Bengali comments

The official labels were hidden from participants.

The official test files are therefore not redistributed in this repository.

---

## 7. Data Redistribution

This repository contains code and derived summary results rather than a redistribution of the complete third-party datasets.

Before downloading, redistributing, or publishing any dataset, users should check its original license and the rules of the AISoMe 2026 shared task.

---

## 8. Important Reproducibility Note

The paper reports that the exact sampling procedure used to construct the final Samanantar-derived parallel dataset was not recorded in sufficient detail.

Therefore, rebuilding the exact original NMT training corpus from this repository alone may not be possible.

Similarly, the original NMT optimizer state was not retained when training was resumed across Kaggle sessions.
