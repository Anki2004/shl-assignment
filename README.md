# Grammar Scoring Engine for Spoken Audio

Predicts a continuous grammar score (0-5) from a 45-60 second spoken audio clip.
Built for the `shl-hiring-assessment-2026` Kaggle competition (769 training clips, 216 test clips).

**Result:** 5-fold CV RMSE **0.50** (predicting the mean gives about 1.24).

## Approach

Only 769 labeled clips are available, so no large model is trained from scratch. Pretrained models act as frozen feature extractors, and small models are fitted on top, which limits overfitting.

```
audio -> Whisper transcript -> handcrafted features -> LightGBM  \
audio -> WavLM embeddings   -> Ridge, SVR                         >-> weighted stack -> score (0-5)
transcript -> DeBERTa-v3 fine-tune                               /
```

1. **Transcription.** Whisper-medium (faster-whisper) with a filler-word prompt, so "um", "uh" and repeated words stay in the text. A cleaned transcript would hide the grammar errors we want to measure. Word timestamps and word confidences are saved.
2. **Handcrafted features.** Speech rate, pause counts and lengths, filler and repetition rates, vocabulary variety, ASR confidence, detected-language probability, and GPT-2 perplexity as a fluency proxy. Model: LightGBM.
3. **Audio embeddings.** WavLM-base-plus, all 13 hidden layers, mean and standard deviation over time. Models: Ridge (alpha 1000) and PCA(256) + SVR (C=10).
4. **Text model.** DeBERTa-v3-base fine-tuned on the transcripts as a regressor (mean pooling, linear head, MSE loss).
5. **Ensemble.** Non-negative least squares on out-of-fold predictions, then clipping to [0, 5].

## Results

5-fold cross-validation, stratified by score, fixed seed, identical folds for every model.

| Model | OOF RMSE |
|---|---|
| Predict the mean | about 1.24 |
| Handcrafted features (LightGBM) | 0.811 |
| DeBERTa on transcripts | 0.88 - 0.89 |
| WavLM Ridge | 0.530 |
| WavLM PCA + SVR | 0.530 |
| **Final stack** | **0.502** |

The stack scores 0.505 on clips with label above 0.

## Key findings

- Audio carries most of the signal. Text adds a smaller gain.
- Whisper confidence (`prob_mean`) is the strongest handcrafted feature (correlation 0.62 with the score).
- The 37 zero-score training clips come from a separate, louder recording source (filenames `audio_50xx`) that does not appear in the test set. Dropping them from training changed nothing (0.5432 vs 0.5424), so they stay.
- `sample_submission.csv` does not match the test list (only 25 of its 204 names are in `test.csv`), so the submission covers all 216 rows of `test.csv`.
- Train and test folders contain files with the same names. Always load with the split folder in the path.

## Limitations

- Stack weights are fitted on the same out-of-fold predictions, and the best DeBERTa epoch was picked on each validation fold. Both make the CV score slightly optimistic.
- Fold-to-fold spread is larger than the differences between the final model variants.
- Whisper can silently correct some grammar errors, which caps the text model.

## Reproduce

Runs on Kaggle with 2x Tesla T4.

```bash
pip install -r requirements.txt
```

1. Download the competition data (not included in this repo).
2. Run the notebook in order: transcription, WavLM embeddings, handcrafted features, baseline CV, DeBERTa folds, stack.
3. The final cell writes `submission.csv` with columns `filename,label` (216 rows).

Intermediate files (`transcripts/`, `wavlm_*.npy`, `hand_*.parquet`, `oof_*.npy`) are generated and not committed.

## Notes from development

- `faster-whisper` failed with an old PyAV version on Kaggle. Fix: read audio with `soundfile` and pass a numpy array.
- DeBERTa loaded in fp16 and `GradScaler` rejected the gradients. Fix: call `.float()` on the model.
- Two DeBERTa runs shared one variable name, which briefly duplicated a stack column. Stack inputs are now loaded from files by name.

## Possible improvements

- Larger audio encoder (WavLM-large).
- An ASR model that keeps verbatim speech.
- A grammar-error-detection model trained on the transcripts.
