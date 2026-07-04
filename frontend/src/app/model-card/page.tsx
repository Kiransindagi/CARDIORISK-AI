export default function ModelCard() {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Model Card</h1>
      <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 prose max-w-none">
        <h2>Intended Use</h2>
        <p>Educational and portfolio demonstration of structured-data ML, explainability, threshold optimization, API deployment, and ML engineering. This is <strong>not</strong> intended for medical diagnosis.</p>
        <h2>Dataset</h2>
        <p>UCI Cleveland Heart Disease dataset (303 observations, 13 predictive features).</p>
        <h2>Methodology</h2>
        <p>80/20 stratified development/test split. Evaluated using Repeated Stratified K-Fold CV. Final model is an uncalibrated Logistic Regression with balanced class weighting.</p>
        <h2>Limitations</h2>
        <ul>
          <li>Small historical dataset, introducing possible distribution shift.</li>
          <li>Limited representativeness (narrow demographic spread).</li>
          <li>Subgroup analysis is constrained by extremely small subgroup sizes.</li>
          <li>No external or prospective validation.</li>
        </ul>
      </div>
    </div>
  );
}
