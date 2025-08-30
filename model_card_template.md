# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Task:** Binary classification — predict `salary` (>50K vs <=50K).
- **Algorithm:**  RandomForestClassifier
## Intended Use
- **Primary use:** Educational ML pipeline (D501) demonstrating training, evaluation, and slice analysis.
- **Out-of-scope uses:** Any high-stakes decisions (employment, credit, housing).
## Training Data
- **Source:** `data/census.csv` (Adult Census Income dataset variant).
- **Target label:** `salary`
- **Categorical features:** `workclass, education, marital-status, occupation, relationship, race, sex, native-country`
- **Split:** 80/20 train/test (stratified), `random_state=42`.
## Evaluation Data

## Metrics
_Please include the metrics used and your model's performance on those metrics._
- **Precision:** `0.7417`
- **Recall:** `0.6154`
- **F1:** `0.6727`
- **Thresholding:** default classifier threshold (prob>0.5 where applicable).
## Ethical Considerations
- Dataset contains sensitive attributes (e.g., sex, race); use slice metrics to monitor disparities.
- This is a course project; performance and calibration may not generalize.
## Caveats and Recommendations
- Re-train if data distribution shifts.
- Track performance drift with periodic re-evaluation on fresh samples.