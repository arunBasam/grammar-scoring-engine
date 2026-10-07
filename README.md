# Grammar Scoring Engine for Spoken Audio

A machine-learning pipeline for predicting a continuous **English grammar score from 0–5** using spoken-audio recordings.

This project was developed for the **SHL Hiring Assessment 2026 – Grammar Scoring Engine** competition.

## 🎯 Objective

The goal is to predict a continuous grammar/MOS score from approximately **45–60 second spoken-audio clips**.

The dataset contains:

- **769 training audio clips**
- **216 test audio clips**
- Continuous target scores from **0–5**
- Evaluation metrics: **RMSE** and **Pearson correlation**

The main challenge is that grammar quality is reflected not only in the words being spoken, but also in fluency, pauses, disfluencies, and other characteristics of spoken language.

---

## 🧠 Approach

The final v4 pipeline combines multiple views of the same speech recording.

### 1. Hand-crafted linguistic and prosodic features

Features include:

- Sentence length statistics
- Lexical diversity
- Word-length statistics
- Fillers and repetitions
- Conjunction and subordination ratios
- Auxiliary-verb usage
- Past-tense and `-ing` ratios
- Speech rate
- Pause statistics
- Speech/silence ratio
- ASR confidence
- GPT-2 language-model features
- Grammar-error-correction edit ratio

These features are used with tree-based regression models.

### 2. Text representations

Audio is transcribed using Whisper.

The transcript is then represented using:

- RoBERTa embeddings
- GPT-2 language-model features
- Hand-crafted linguistic features

### 3. Audio representations

Several pretrained speech encoders are explored:

- wav2vec2
- Whisper-small
- Whisper-medium
- Whisper-large-v3
- WavLM-large

Layer-wise representations are evaluated to identify the most useful encoder layers.

### 4. Attention-pooling neural model

The final pipeline also includes an attention-based neural head over frame-level Whisper representations.

Instead of reducing the complete recording to a single mean vector, the model learns which temporal portions of the speech are more useful for predicting the grammar score.

The architecture uses:

- Layer normalization
- Linear projection
- Transformer encoder layer
- Attention pooling
- Mean pooling
- MLP regression head

### 5. Ensemble / stacking

Predictions from the different models are combined using **non-negative least-squares stacking**.

The final stack is selected using nested cross-validation, with optional linear calibration.

Predictions are constrained to the valid **0–5** score range.

---

## 🔬 Validation

The pipeline uses repeated cross-validation and out-of-fold predictions.

The final stacker is additionally evaluated using **nested cross-validation** to obtain a more realistic estimate of generalization performance.

The notebook reports:

- Training RMSE
- OOF RMSE
- OOF Pearson correlation
- Nested-CV RMSE
- Nested-CV Pearson correlation
- Individual model/block performance

> The training RMSE is expected to be optimistic because the model is evaluated on data used during training. The nested-CV result is the more realistic estimate of performance on unseen data.

---

## 📊 Key Findings

The experiments showed that **audio encoder representations provided stronger signals than transcript-only representations**.

This is likely because automatic speech recognition can remove or normalize some disfluencies and small grammatical errors that are present in the original speech.

To recover some of this information, the final pipeline incorporates:

- Verbatim-style transcription
- ASR token-confidence information
- Prosodic features
- Pause statistics
- Grammar-correction edit ratios
- Multiple speech-encoder representations

Combining these complementary representations through stacking provides a stronger final model than relying on a single feature block.

---

## 🏗️ Pipeline

```text
                    Spoken Audio
                         │
                         ▼
                16 kHz Mono Audio
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
     Whisper ASR                Audio Encoders
          │                             │
          ▼                  ┌──────────┼──────────┐
   Transcript Features       │          │          │
          │               wav2vec2   Whisper     WavLM
          │                          / Whisper-large
          ▼                             │
    ┌───────────────┐                   │
    │ Linguistic    │                   │
    │ + Prosodic    │                   │
    │ + LM Features │                   │
    └───────┬───────┘                   │
            │                            │
            └────────────┬───────────────┘
                         ▼
                 Multiple Regressors
                         │
                         ▼
                Out-of-Fold Predictions
                         │
                         ▼
              Nested-CV Stacking
                         │
                         ▼
                  Final Score 0–5
