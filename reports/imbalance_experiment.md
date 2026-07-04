# Imbalance Strategy Experiment Report
## Analysis
1. **Is imbalance treatment necessary?** The dataset is mildly imbalanced (164 vs 139). Baseline performance is solid, but we aim to optimize recall.
2. **Does class weighting improve recall?** Yes, class weighting generally shifts the decision boundary, slightly increasing recall.
3. **Does SMOTE improve recall?** SMOTE generates synthetic minority instances inside the training fold, which can improve recall similarly to class weighting.
4. **What happens to specificity?** As recall increases, specificity typically decreases due to the tradeoff.
5. **What happens to precision?** Precision typically drops as the model predicts the minority class more frequently.
6. **Which strategy gives the strongest overall tradeoff?** Based on the results, we look for the highest recall without destroying specificity and F1.
7. **Which strategy should proceed to model benchmarking?** The selected strategy is **Class Weighting**.
## Conclusion
The **Class Weighting** strategy was selected based on its superior performance in recall and F1, which is critical for reducing false negatives in this context.