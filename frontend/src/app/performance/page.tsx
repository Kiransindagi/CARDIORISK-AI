export default function Performance() {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Model Performance Metrics</h1>
      <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
        <p className="mb-4">This section displays actual project results evaluated on a strictly held-out test set.</p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-slate-50 p-4 rounded-lg border text-center">
            <div className="text-sm text-slate-500">Accuracy</div>
            <div className="text-2xl font-bold">88.5%</div>
          </div>
          <div className="bg-slate-50 p-4 rounded-lg border text-center">
            <div className="text-sm text-slate-500">Recall</div>
            <div className="text-2xl font-bold">96.4%</div>
          </div>
          <div className="bg-slate-50 p-4 rounded-lg border text-center">
            <div className="text-sm text-slate-500">Specificity</div>
            <div className="text-2xl font-bold">81.8%</div>
          </div>
          <div className="bg-slate-50 p-4 rounded-lg border text-center">
            <div className="text-sm text-slate-500">ROC-AUC</div>
            <div className="text-2xl font-bold">0.957</div>
          </div>
        </div>
        <h3 className="font-semibold text-lg mb-2">Threshold Tradeoff</h3>
        <p className="text-sm text-slate-600">The decision threshold was strictly optimized on development probabilities to prioritize recall (minimizing costly false negatives), resulting in an optimized threshold of 0.42.</p>
      </div>
    </div>
  );
}
