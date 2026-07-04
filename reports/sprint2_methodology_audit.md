# Sprint 2 Methodology Audit

## Issue Found
During the pre-sprint audit, it was discovered that the Sigmoid-calibrated Logistic Regression model was erroneously selected over the uncalibrated Logistic Regression model. The uncalibrated Logistic Regression model achieved a lower Brier score (0.119) compared to the calibrated model (0.123). A lower Brier score indicates better probability calibration.

## Audit Procedure
1. Verified that the Brier scores were generated using identical out-of-fold predictions.
2. Verified that the calibration evaluation was leakage-safe using 5-fold CV.
3. Verified the actual reported values.
4. Assessed whether the decision threshold of 0.42 was optimized using the probabilities from the incorrectly saved calibrated architecture.

## Result
The reported Brier scores were correct. Sigmoid calibration worsened the Brier score, likely because Logistic Regression natively produces well-calibrated probabilities, and the small dataset size led to minor overfitting during calibration. 

## Corrective Actions
- **Model Changed**: Yes. The final production architecture was changed from `CalibratedClassifierCV(LogisticRegression)` to standard `LogisticRegression`.
- **Threshold Changed**: The decision threshold optimization was rerun using the uncalibrated probabilities to ensure complete alignment.
- **Final Test Evaluation Required Correction**: Yes. The final held-out test evaluation was re-run using the updated model and corresponding optimized threshold to prevent any methodological inconsistency.
- **Final Production Model Architecture**: `LogisticRegression(class_weight='balanced', C=...)` [Uncalibrated].
