# Reproduction Guide

## 1. Environment

The original experiments were performed in Kaggle notebooks.

The paper records:

* Python 3.12.13
* Kaggle Docker image version 28755
* PyTorch
* Transformers
* SentencePiece
* sacreBLEU
* scikit-learn

The exact package versions were not separately recorded.

---

## 2. Pipeline

The complete pipeline consists of:

```text
Parallel NMT data
        |
        v
Train multilingual Transformer
        |
        v
Epoch-19 NMT checkpoint
        |
        v
Translate English labelled data
        |
        +-------------------+
        |                   |
        v                   v
      Hindi              Bengali
        |                   |
        +---------+---------+
                  |
                  v
             Pool data
                  |
                  v
       Train stance classifiers
                  |
          +-------+-------+
          |       |       |
          v       v       v
       Model 1 Model 2 Model 3
          |       |       |
          +-------+-------+
                  |
                  v
              Inference
                  |
                  v
        Hindi/Bengali predictions
```

---

## 3. NMT Training

The NMT model uses a standard encoder-decoder Transformer.

### Architecture

* 6 encoder layers
* 6 decoder layers
* model dimension: 512
* attention heads: 8
* feed-forward dimension: 2048
* dropout: 0.1
* approximately 68.8 million trainable parameters
* shared encoder/decoder embeddings
* tied output projection
* sinusoidal positional encoding

### Optimisation

* Adam
* beta values: `(0.9, 0.98)`
* epsilon: `1e-9`
* label smoothing: `0.1`
* inverse-square-root learning-rate schedule
* 8,000 warm-up updates
* gradient clipping: 1.0
* fp16
* seed: 42
* 20 epochs
* 153,120 total updates

---

## 4. NMT Validation

The final retained checkpoint was the epoch-19 checkpoint.

Validation BLEU:

| Epoch | Hindi | Bengali | Average |
| ----- | ----: | ------: | ------: |
| 17    | 23.65 |   15.93 |   19.79 |
| 18    | 23.95 |   16.46 |   20.21 |
| 19    | 25.33 |   16.35 |   20.84 |

The validation scores were calculated using a small fixed sample of 50 pairs per language.

The BLEU evaluation used beam search.

However, the actual stance-data translation used greedy decoding.

---

## 5. Classifier Training

### Model 1

```text
Encoder: google/muril-base-cased
Split: instance-level stratified 90/10
Seed: 42
Learning rate: 2e-5
Epochs: 4
```

This split can place the Hindi and Bengali translations of the same English source sentence on opposite sides of the split.

---

### Model 2

```text
Encoder: google/muril-base-cased
Split: source-grouped stratified 90/10
Seed: 7
Learning rate: 2e-5
Epochs: 4
```

Both translations of an English source sentence are kept in the same split.

---

### Model 3

```text
Encoder: xlm-roberta-base
Split: source-grouped stratified 90/10
Seed: 7
Learning rate: 1.5e-5
Epochs: 5
```

---

## 6. Classifier Settings

All three models used:

```text
Maximum sequence length: 256
Training batch size/device: 16
Evaluation batch size/device: 32
Number of GPUs: 2
Precision: fp16
Loss: cross entropy
Class weighting: balanced
Selection metric: macro-F1
```

---

## 7. Running the Classifiers

Model 1:

```bash
python src/12_train_run.py \
    --model google/muril-base-cased \
    --tag run1_muril_leaky \
    --seed 42 \
    --epochs 4 \
    --lr 2e-5
```

Model 2:

```bash
python src/12_train_run.py \
    --model google/muril-base-cased \
    --tag run2_muril_group \
    --seed 7 \
    --epochs 4 \
    --lr 2e-5 \
    --group_split
```

Model 3:

```bash
python src/12_train_run.py \
    --model xlm-roberta-base \
    --tag run3_xlmr_group \
    --seed 7 \
    --epochs 5 \
    --lr 1.5e-5 \
    --group_split
```

---

## 8. Inference

After training, run:

```bash
python src/10_run_inference.py
```

The inference stage produces predictions for the Hindi and Bengali test data.

---

## 9. Reproducibility Limitations

Exact bit-for-bit reproduction of the original NMT model is not guaranteed.

The NMT training was resumed over multiple Kaggle sessions and the optimizer state was not preserved when training resumed.

The exact sampling procedure used to construct the final Samanantar-derived parallel dataset was also not fully recorded.

Consequently, this repository should be understood as a reproduction-oriented implementation and documentation of the reported pipeline rather than a guarantee of identical NMT weights.

---

## 10. Preliminary Experiments

The following scripts document earlier experiments:

```text
src/archive/09_train_stance_classifier.py
src/archive/11_train_v3.py
src/archive/11_compare_translators.py
```

These experiments were not used to generate the final three submitted models.

The final submitted configurations were produced by:

```text
src/12_train_run.py
```
