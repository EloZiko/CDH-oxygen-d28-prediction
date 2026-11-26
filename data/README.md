# Data

The patient data of the article are **not** in this repository, for privacy
reasons. Researchers who would like to work with them should contact the
corresponding author of the article, Edouard Leyne, to discuss a collaboration
and its conditions.

To use your own data, put a CSV file in this folder. Git ignores every file
here except this one, so patient data cannot be published by mistake.

## Expected format

A CSV file (comma-separated), one row per fetus.

**Predictors.** One column for each of the four predictors of the model:

| Column             | What it is                                   | Unit or values  |
| ------------------ | -------------------------------------------- | --------------- |
| `oe_lhr`           | o/e lung-to-head ratio (ultrasound)          | %               |
| `liver_herniation` | Liver herniated into the thorax (ultrasound) | 1 = yes, 0 = no |
| `dvp`              | Deepest amniotic fluid pocket (ultrasound)   | mm              |
| `oe_tflv`          | o/e total fetal lung volume (MRI)            | %               |

o/e: observed-to-expected.

**Outcome.** A fifth column, `oxygen_d28`: supplemental oxygen at day 28 of
life (`1` = yes, `0` = no).

- Use the measurements of the referral centre work-up, as in the article.
- Write percentages from 0 to 100: `35.5`, not `0.355`.
- Leave a cell empty when the value is missing.
- Other columns are ignored.
- As in the article, a fetus is used only if its outcome is known and at least
  2 of the 4 predictors are filled in.

## Example

Two fictitious fetuses. The second one has no `dvp` value (empty cell):

```csv
oe_lhr,liver_herniation,dvp,oe_tflv,oxygen_d28
32.0,1,75,28.5,1
48.0,0,,41.0,0
```

## Cohort files of the article

[`analysis/1_clean_data.py`](../analysis/1_clean_data.py) writes one file per
centre from the raw databases, in the same format, with more columns. The 16
candidate variables, in this order:

| Column                   | What it is                                    | Unit or values  |
| ------------------------ | --------------------------------------------- | --------------- |
| `parity`                 | Number of previous deliveries                 | 0, 1, 2...      |
| `maternal_weight`        | Maternal weight                               | kg              |
| `maternal_height`        | Maternal height                               | m               |
| `ivf`                    | Conception by in vitro fertilisation          | 1 = yes, 0 = no |
| `fetal_sex`              | Fetal sex                                     | 0 or 1          |
| `oe_lhr`                 | o/e lung-to-head ratio (ultrasound)           | %               |
| `liver_herniation`       | Liver herniated into the thorax (ultrasound)  | 1 = yes, 0 = no |
| `dvp`                    | Deepest amniotic fluid pocket (ultrasound)    | mm              |
| `cervical_length`        | Cervical length (ultrasound)                  | mm              |
| `placental_localisation` | Placenta (ultrasound)                         | see below       |
| `other_malformations`    | Other anomaly, including IUGR < 5th centile   | 1 = yes, 0 = no |
| `cardiac_anomaly`        | Cardiac anomaly                               | 1 = yes, 0 = no |
| `genetic_anomaly`        | Abnormal genetic testing                      | 1 = yes, 0 = no |
| `cgh_exome_anomaly`      | Abnormal CGH array or exome                   | 1 = yes, 0 = no |
| `mri_performed`          | Fetal MRI performed                           | 1 = yes, 0 = no |
| `oe_tflv`                | o/e total fetal lung volume (MRI)             | %               |

Placenta: 1 = anterior, 2 = posterior, 3 = fundal. In the Leuven database,
parity 3 means 3 or more. IUGR: intrauterine growth restriction.

Then the outcome `oxygen_d28`, and four columns that only describe the
cohorts (Table 1): `ga_first_us_referral` (gestational age at the first
ultrasound in the referral centre, weeks), `cardiac_scan_performed` and
`exome_panel_performed` (1 = yes, 0 = no), and `feto` (fetal endoscopic
tracheal occlusion, 1 = yes, 0 = no).
