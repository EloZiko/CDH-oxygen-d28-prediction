# Data

The patient data of the article are **not** in this repository, for privacy
reasons. For collaboration or research purposes, they can be requested from
the corresponding author of the article.

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
