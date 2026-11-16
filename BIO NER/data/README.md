# Dataset Folder

Place your labelled biomedical NER dataset here in **CoNLL format**.

### Format
- One token and its BIO label per line, separated by a space
- Blank line between sentences

### Example
```
The O
patient O
was O
diagnosed B-DISEASE
with O
type O
2 O
diabetes I-DISEASE
mellitus I-DISEASE
. O

The O
patient O
is O
taking B-DRUG
metformin I-DRUG
. O
```

### Sample file included
`sample_biomedical_ner.conll` contains a small synthetic biomedical dataset so you can run the training and GUI immediately. Replace it with a larger real-world corpus (e.g., from NCBI Disease, BC5CDR, or i2b2) for better results.
