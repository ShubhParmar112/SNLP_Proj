## Task C - ABSA tuple extraction (test split)

Test split, n=900, seed=42. 95% bootstrap CI over 1000 resamples.

| Model | micro tuple-F1 | 95% CI | Precision | Recall | Polarity acc. on detected |
|---|---|---|---|---|---|
| Majority class | 0.216 | [0.196, 0.238] | 0.310 | 0.166 | 0.567 |
| Stratified random | 0.147 | [0.130, 0.165] | 0.149 | 0.144 | 0.413 |
| Lexicon + LogReg polarity | 0.332 | [0.307, 0.356] | 0.379 | 0.296 | 0.625 |
| TF-IDF LogReg (full baseline) | 0.452 | [0.427, 0.476] | 0.432 | 0.472 | 0.612 |

## Task A - document sentiment

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority class | 0.507 | 0.224 |
| TF-IDF + LogReg | 0.750 | 0.722 |