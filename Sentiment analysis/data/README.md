# Dataset Folder – Sentiment Analysis

## Dataset Format
Place your labelled sentiment dataset here as a CSV file named `sample_imdb.csv`.

### Required Columns
| Column | Description |
|--------|-------------|
| `text` | The review text |
| `label` | Sentiment label: `Positive`, `Negative`, or `Neutral` |

### Example
```
text,label
"The movie was fantastic and I loved every minute of it.",Positive
"Terrible plot, terrible acting, complete waste of time.",Negative
"It was okay, nothing special but watchable.",Neutral
```

## Included Sample
A small synthetic sample (`sample_imdb.csv`) is included so the project runs immediately. Replace it with the real IMDb Movie Reviews dataset for better results.

**Suggested dataset:** IMDb Movie Reviews (https://www.imdb.com/) or any labelled review corpus.
