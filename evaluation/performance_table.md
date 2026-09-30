| Metric | Trained on CPU | Trained on GPU | On edge device |
|---|---|---|---|
| Age MAE (years, continuous) | 7.57 | 7.73 | -- |
| Age bucket accuracy (adult/elderly) | 92.87% | 92.53% | 85.00% |
| Gender accuracy | 86.80% | 87.14% | 65.00% |
| Expression accuracy | 57.79% | 56.42% | 60.00% |
| Training time (seconds, age/gender model) | 2905.00 | 317.27 | -- |
| Training time (seconds, expression model) | 7833.92 | 670.76 | -- |
| Face detection latency (ms) | -- | -- | 576.80 |
| Age/gender inference latency (ms) | -- | -- | 219.70 |
| Expression inference latency (ms) | -- | -- | 6394.60 |
