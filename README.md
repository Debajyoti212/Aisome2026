# Squad4 at AISoMe 2026

## Translate-Train Climate-Change Stance Detection in Hindi and Bengali

This repository contains the code, experimental configurations, results, and reproduction instructions for **Squad4's participation in the AISoMe 2026 track of FIRE 2026**.

Our task was to classify Hindi and Bengali social-media comments according to their stance towards the claim:

> **"Climate change and global warming are serious concerns."**

The three stance labels are:

* `Favour`
* `Against`
* `None`

Because we did not have labelled Hindi or Bengali training data, we used a **translate-train** approach.

---

## System Overview

Our system consists of three main stages.

```text
English labelled data
        |
        v
Multilingual Transformer NMT
        |
        +------------------+
        |                  |
        v                  v
     Hindi              Bengali
        |                  |
        +---------+--------+
                  |
                  v
       Translated training data
                  |
          +-------+-------+
          |       |       |
          v       v       v
        Model 1 Model 2 Model 3
        MuRIL   MuRIL   XLM-R
          |       |       |
          +-------+-------+
                  |
                  v
       Hindi/Bengali predictions
```

---

## Main Approach

We trained one multilingual Transformer neural machine translation model from scratch using:

* 980,000 English-Hindi sentence pairs
* 980,000 English-Bengali sentence pairs
* 48,000-piece SentencePiece vocabulary
* 6 encoder layers
* 6 decoder layers
* model dimension 512
* 8 attention heads
* feed-forward dimension 2048
* 20 training epochs
* Adam optimisation
* label smoothing of 0.1
* mixed-precision training
* random seed 42

The final NMT checkpoint achieved:

| Language |  BLEU |
| -------- | ----: |
| Hindi    | 25.33 |
| Bengali  | 16.35 |
| Average  | 20.84 |

The NMT model was then used to translate 3,008 English labelled comments into Hindi and Bengali, producing 6,016 translated training examples.

---

## Classifier Models

We trained three classifier configurations.

| Model   | Encoder                   | Split                           | Seed | Learning Rate | Epochs |
| ------- | ------------------------- | ------------------------------- | ---: | ------------: | -----: |
| Model 1 | `google/muril-base-cased` | Instance-level stratified 90/10 |   42 |          2e-5 |      4 |
| Model 2 | `google/muril-base-cased` | Source-grouped stratified 90/10 |    7 |          2e-5 |      4 |
| Model 3 | `xlm-roberta-base`        | Source-grouped stratified 90/10 |    7 |        1.5e-5 |      5 |

All three classifiers used:

* maximum sequence length: 256 tokens
* 16 training examples per device
* 32 evaluation examples per device
* 2 GPUs
* cross-entropy loss
* balanced class weights
* fp16 precision
* macro-F1 for model selection

---

## Results

### Official Results

| Language | Model 1 | Model 2 | Model 3 |
| -------- | ------: | ------: | ------: |
| Bengali  |  0.4699 |  0.4756 |  0.5274 |
| Hindi    |  0.4222 |  0.4288 |  0.4561 |

The official results were supplied by the AISoMe 2026 organisers.

The paper reports that the best submitted run was ranked 12th of 18 teams for Bengali and 17th of 18 teams for Hindi.

---

### Internal Validation Results

| Model   | Overall Macro-F1 | Bengali Macro-F1 | Hindi Macro-F1 | Accuracy |
| ------- | ---------------: | ---------------: | -------------: | -------: |
| Model 1 |           0.6533 |           0.6152 |         0.6890 |   0.6545 |
| Model 2 |           0.6596 |           0.6617 |         0.6574 |   0.6628 |
| Model 3 |           0.6817 |           0.6670 |         0.6964 |   0.6827 |

These validation results are from our own held-out validation splits and should not be treated as directly comparable to the official competition scores.

---

## Important Experimental Note

The three models are **configuration-level comparisons rather than controlled ablations**.

Model 1 and Model 2 differ in both:

* validation split protocol
* random seed

Model 2 and Model 3 differ in:

* encoder
* learning rate
* number of epochs

Therefore, the experiments do not isolate the effect of any single factor.

In particular, the results should not be interpreted as a controlled experiment showing that XLM-R is better than MuRIL.

---

## Repository Structure

```text
aisome2026-squad4/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── train_nmt.py
│   ├── 08_translate_training_csv.py
│   ├── 10_run_inference.py
│   ├── 12_train_run.py
│   └── archive/
│       ├── 09_train_stance_classifier.py
│       ├── 11_train_v3.py
│       └── 11_compare_translators.py
│
├── notebooks/
│   └── aisome2026_squad4.ipynb
│
├── configs/
│   ├── run1_muril_leaky.yaml
│   ├── run2_muril_group.yaml
│   └── run3_xlmr_group.yaml
│
├── results/
│   ├── official_results.csv
│   ├── internal_validation_results.csv
│   └── nmt_validation_results.csv
│
├── examples/
│   └── sample_translations.csv
│
└── docs/
    ├── reproduction.md
    └── data.md
```

---

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The original experiments were performed in a Kaggle environment using Python 3.12.13 and the package versions available in the corresponding Kaggle image.

Exact package versions were not separately recorded during the original experiments.

---

## Reproduction Pipeline

### Step 1: Prepare the NMT data

Place the parallel English-Hindi and English-Bengali data according to the instructions in:

```text
docs/data.md
```

---

### Step 2: Train the multilingual NMT model

```bash
python src/train_nmt.py
```

The NMT model uses the `<hi>` and `<bn>` source-side language tags to determine the target language.

---

### Step 3: Translate the labelled English data

```bash
python src/08_translate_training_csv.py
```

This produces Hindi and Bengali versions of the English labelled training comments.

---

### Step 4: Train the classifier

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

## Step 5: Run Inference

Use:

```bash
python src/10_run_inference.py
```

The inference script applies the trained classifiers to the Hindi and Bengali test comments.

---

## Data

We used English labelled data for stance classification and Samanantar-derived parallel data for NMT training.

The classifier data contained:

* 3,008 English labelled comments
* 3,008 Hindi translations
* 3,008 Bengali translations
* 6,016 translated classifier examples after pooling

The official test sets contained:

* 500 Hindi comments
* 500 Bengali comments

The official test labels were hidden by the organisers.

For licensing and redistribution reasons, the repository does not include the complete original datasets or official hidden-label test data.

See:

```text
docs/data.md
```

for data provenance and preparation instructions.

---

## Reproducibility

The retained scripts corresponding to the final pipeline are:

* `08_translate_training_csv.py`
* `10_run_inference.py`
* `12_train_run.py`

The repository also contains preliminary scripts in:

```text
src/archive/
```

These scripts document experiments that were performed but were not used to produce the final submitted models.

The original NMT training was resumed across multiple Kaggle sessions without preserving optimizer state. Consequently, exact bit-for-bit reproduction of the original NMT weights is not guaranteed.

---

## Limitations

The main limitations identified in the study include:

1. Machine translation errors can alter or reverse stance-bearing meaning.
2. Model 1's instance-level split can place Hindi and Bengali translations of the same English source on different sides of the train/validation split.
3. The three submitted configurations do not form clean single-factor ablations.
4. The internal validation data consist of machine-translated text rather than native Hindi/Bengali comments.
5. There is a substantial gap between internal validation scores and official test performance.
6. The NMT model had not fully converged when training ended.
7. The NMT optimizer state was not retained across Kaggle sessions.
8. The NLLB comparison performed during the study used a preliminary classifier and therefore does not constitute a controlled classifier comparison.

---

## Citation

If you use this repository, please cite the corresponding AISoMe 2026 working note.

```bibtex
@inproceedings{majmuder2026aisome,
  title     = {Squad4 at AISoMe 2026: Translate-Train Climate-Change Stance Detection in Hindi and Bengali using a Multilingual Transformer NMT, MuRIL and XLM-R},
  author    = {Majumder, Debajyoti and Mandol, Anayan},
  booktitle = {Working Notes of FIRE 2026},
  year      = {2026},
  note      = {AISoMe 2026 Track}
}
```

---

## Team

**Squad4**

* Debajyoti Majumder
* Anayan Mandol

B.Tech Computer Science and Engineering  
Amity University Kolkata  
West Bengal, India

---

## Acknowledgements

We thank the AISoMe 2026 organisers for the shared task and evaluation.

All experiments were conducted using Kaggle notebooks.
