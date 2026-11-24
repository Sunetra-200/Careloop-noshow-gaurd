# CareLoop Clinics — No-Show Guard

**GitHub Repository:** `https://github.com/Sunetra-200/Careloop-noshow-gaurd`

## 4-Line Summary
1. **Part 1 (Data Foundations):** Cleaned 1,020 raw bookings down to 990 valid appointments (218 no-shows, **22.0%**) and 8 clinics, imputing 17 missing/invalid ages (median 48) and 20 missing distances (median 8.25 km).
2. **Part 2 (SQL Analysis):** No-shows cluster at **Gomti Nagar Care Point** (**33.3%**, 90 appointments) and **Hadapsar Care Centre** (**27.3%**, 143 appointments), and climb sharply with booking lead time from **16.2%** (1–7 days) to **60.5%** (31+ days).
3. **Part 3 (Machine Learning):** A **Logistic Regression** pipeline won 5-fold cross-validation (**0.77** CV ROC-AUC; **0.789** test ROC-AUC) and, at a **0.35** threshold, flagged **49** of 198 test bookings with **57.1%** recall and **57.1%** precision.
4. **Part 4 (GenAI Follow-up Review):** Running `04_genai.ipynb` in Colab classifies the first 10 call-centre notes (9 non-blank + 1 `skipped_blank`) via Gemini Flash and cross-tabulates `root_cause` against `reminder_sent` (update this line with your exact `notes_analysed.csv` count after running Colab).

## Business Recommendation
CareLoop Clinics should immediately deploy the **Logistic Regression** risk model at a **0.35** alert threshold every evening to flag high-risk next-day bookings for front-desk phone calls, while tightening automated reminder coverage and long-lead follow-ups. Across the 990 cleaned appointments, **22.0%** (218 visits) were missed, with bookings made **31+ days** ahead missing at **60.5%** (versus **16.2%** for 1–7 days) and **Gomti Nagar Care Point** (**33.3%**) and **Hadapsar Care Centre** (**27.3%**) leading clinic-level no-show rates. In the fitted model, sending a reminder is associated with **69% lower odds** of a no-show (odds ratio **0.31**, holding other features constant), and flagging at **0.35** captures **57.1%** of test-set no-shows across **49** calls (about a quarter of the schedule). Prioritising evening reminder calls to these flagged patients—starting with long-lead bookings, unreminded appointments, and the highest-rate clinics—offers the fastest way to recover empty doctor slots without overwhelming front-desk capacity.

## Model Choice & Threshold
- **Chosen Model:** `Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])`
- **Why:** It achieved the highest 5-fold cross-validated ROC-AUC on the 792-row training set (**~0.77**, ahead of the depth-3 Decision Tree at ~0.68 and default Random Forest at ~0.75) and delivered **0.789** ROC-AUC on the held-out 198-row test set.
- **Chosen Threshold:** **0.35** (selected on training out-of-fold probabilities where it improved F1 from ~0.35 to ~0.49 and recall from ~0.25 to ~0.47 over the 0.50 default). A missed warning costs an empty doctor slot, whereas an extra call costs only a minute.

## Limitations
1. **Pre-split cleaning leakage:** The medians, quartiles, and IQR outlier fences in Task 1.3 were computed on all 990 rows before the train/test split; in production, these statistics must be fit on training rows only.
2. **Patient overlap & random split:** The same `patient_id` can appear in both train and test sets under a random split, whereas a real clinic system is trained on past appointments and evaluated on future dates.
3. **Observational data & small note sample:** Associations (such as `reminder_sent` having an odds ratio of 0.31) do not prove causation, and the 10 analysed follow-up notes are an illustrative sample rather than a clinic-wide distribution.

## Repository Contents
- `data/` — Original `appointments.csv`, `clinics.csv`, and `followup_notes.csv`
- `01_ingest_clean.ipynb`, `rejected_rows.json`, `appointments_clean.csv`, `clinics_enriched.csv`, `load_data.sql`
- `02_queries.sql`
- `03_model.ipynb`, `test_risk_scores.csv`
- `04_genai.ipynb`, `notes_analysed.csv` *(generated when `04_genai.ipynb` is run in Colab with your Gemini API key)*
- `README.md`
