# SyntheticPain

Synthetic training images and fairness analysis code for:

> Heintz T, Ge Y, Boscardin J, Espejo E, Dundas N, Wallace A, Law T, Sinavsky K, Lartigue A, Veuthey T, Lee S, Cobert J. Using facial images to classify pain behaviors in diverse hospitalized patients: development and validation of a pain classification system leveraging synthetic facial data. Manuscript under review.

We trained a pain behavior classifier only on synthetic faces of hospitalized patients and validated it on human faces from three datasets (SFVAMC, UNBC-McMaster, and Delaware). If you use these images or code, please cite the paper.

## Contents

| Folder | What's in it |
| --- | --- |
| `Heintz Pain Dataset - primary analysis/basic_prompt/` | Primary training set: 9,000 images (4,500 `pain`, 4,500 `nopain`), JPEG |
| `Heintz Pain Dataset - secondary analysis/diversity_prompt/` | Sensitivity analysis training set (diversity-enhanced prompts): 9,002 images (4,501 `pain`, 4,501 `nopain`), PNG |
| `fairness/` | Fairness analysis script and group-level results |

## Synthetic images

- Generated with Imagen 3 (`imagen-3.0-generate-002`): the primary set in February and March 2025, the diversity-enhanced set in August and October 2025.
- Each image is a 224 x 224 RGB face crop, the model's input size.
- Labels come from the prompt (pain or no pain). The images were not scored by human raters.
- Every image was reviewed before training. Images were excluded if they did not show a hospitalized patient in a hospital bed, had gross generation errors, or had clearly abnormal content unrelated to the clinical scenario, including culturally stereotyped attire.

Prompts (other generation settings are in Supplemental Tables 1 and 2 of the paper):

| Set | Prompt |
| --- | --- |
| Primary | Hospitalized patient in hospital bed in [pain / no pain] with camera centered straight on the patients face. |
| Diversity-enhanced | Hospitalized {label} patient in hospital bed in [pain / no pain] with camera centered straight on the patients face. |

In the diversity-enhanced set, `{label}` is one of seven race and ethnicity groups, with 643 pain and 643 no pain images per group. The group code is in each file name (`<id>__<CODE>__CPOT__<timestamp>.png`):

| Code | Label |
| --- | --- |
| AIAN | American Indian or Alaska Native |
| ASIA | Asian |
| BLACK | Black or African American |
| HISPLAT | Hispanic or Latino |
| MENA | Middle Eastern or North African |
| NHPACI | Native Hawaiian or Pacific Islander |
| WHITE | White |

## Model

Not shared. The paper's Methods and Supplemental Methods describe it: a VGG-16 network pretrained for face recognition, with frozen convolutional layers and a small classification head, trained only on the synthetic images above with 8-fold cross-validation. It is a research model and has not been validated for clinical use.

## Fairness analysis

`fairness/fairness_analysis.py` reproduces the skin tone fairness results in the paper (pooled human validation cohort, n = 321, threshold 0.5):

- accuracy, true positive rate, false positive rate, and selection rate with 95% bootstrap CIs
- demographic parity ratios relative to Fitzpatrick type II and Monk 2
- permutation chi-square tests across skin tone groups
- paired comparisons of the primary and diversity-enhanced models

The input is a CSV with one row per participant and columns `y_true` (1 = pain behaviors), `fitzpatrick` (1 to 6), `monk` (1 to 10), `prob_primary`, and `prob_diversity`. The script writes group-level tables only.

```
pip install numpy pandas
python fairness/fairness_analysis.py predictions.csv output_folder
```

The study's prediction file is patient-level and is not shared. `fairness/fairness_results.csv` holds the group-level results it produced (the data behind Figures 3 and 4 and Supplemental Figures 2 and 8).

## Data availability

Human validation images, labels, and predictions are not shared because of patient privacy and institutional policy. The UNBC-McMaster and Delaware datasets are available from their original sources. Model weights and training code are not shared.

## License

Code: MIT (see `LICENSE`). Images: CC BY 4.0.

## Questions

Open an issue, or contact the corresponding author listed in the paper.
