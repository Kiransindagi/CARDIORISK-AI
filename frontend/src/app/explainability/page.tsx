export default function Explainability() {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Explainability (SHAP)</h1>
      <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 prose max-w-none">
        <p>This platform uses SHAP (SHapley Additive exPlanations) to explain the output of the machine learning model.</p>
        <h3>Global Feature Importance</h3>
        <p>Across the dataset, the features most strongly associated with model predictions are:</p>
        <ul>
          <li><strong>Number of Major Vessels</strong></li>
          <li><strong>Thalassemia</strong></li>
          <li><strong>Chest Pain Type</strong></li>
        </ul>
        <h3>Local Explanations</h3>
        <p>In the Risk Assessment tool, you can see individual SHAP values broken down into <strong>Risk-Increasing</strong> and <strong>Risk-Decreasing</strong> factors for a specific patient.</p>
        <div className="bg-amber-50 p-4 rounded text-amber-900 text-sm mt-6">
          <strong>Caution:</strong> Feature contribution does not mean causation. SHAP provides statistical feature attribution, not a biological mechanism.
        </div>
      </div>
    </div>
  );
}
