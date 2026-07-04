'use client';
import { useState } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

export default function RiskAssessment() {
  const [formData, setFormData] = useState({
    age: 50, sex: 'Male', chest_pain_type: 'Typical Angina', resting_bp: 120, cholesterol: 200,
    fasting_blood_sugar: 'No', resting_ecg: 'Normal', max_heart_rate: 150, exercise_induced_angina: 'No',
    st_depression: 0.0, st_slope: 'Flat', num_major_vessels: 0, thalassemia: 'Normal'
  });
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e: any) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value
    }));
  };

  const handleSubmit = async (e: any) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error?.message || 'Prediction failed');
      }
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto w-full transition-all">
      <div className="text-center mb-10">
        <h1 className="text-4xl font-outfit font-bold tracking-tight text-slate-800 mb-3">Clinical Assessment</h1>
        <p className="text-slate-600">Enter patient vitals for real-time risk evaluation</p>
      </div>
      
      {error && <div className="glass-panel bg-red-500/10 border-red-500/20 text-red-700 p-4 mb-8 text-center">{error}</div>}
      
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        <div className="lg:col-span-7 glass-panel p-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 ml-1">Age</label>
                <input type="number" name="age" value={formData.age} onChange={handleChange} className="glass-input w-full p-3 bg-white/40" required />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 ml-1">Sex</label>
                <select name="sex" value={formData.sex} onChange={handleChange} className="glass-input w-full p-3 bg-white/40" required>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>
            </div>
            
            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-700 ml-1">Chest Pain Type</label>
              <select name="chest_pain_type" value={formData.chest_pain_type} onChange={handleChange} className="glass-input w-full p-3 bg-white/40" required>
                <option value="Typical Angina">Typical Angina</option>
                <option value="Atypical Angina">Atypical Angina</option>
                <option value="Non-anginal Pain">Non-anginal Pain</option>
                <option value="Asymptomatic">Asymptomatic</option>
              </select>
            </div>
            
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 ml-1">Resting BP (mmHg)</label>
                <input type="number" name="resting_bp" value={formData.resting_bp} onChange={handleChange} className="glass-input w-full p-3 bg-white/40" required />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 ml-1">Cholesterol (mg/dl)</label>
                <input type="number" name="cholesterol" value={formData.cholesterol} onChange={handleChange} className="glass-input w-full p-3 bg-white/40" required />
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 ml-1">Max Heart Rate</label>
                <input type="number" name="max_heart_rate" value={formData.max_heart_rate} onChange={handleChange} className="glass-input w-full p-3 bg-white/40" required />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 ml-1">Fasting Blood Sugar &gt; 120</label>
                <select name="fasting_blood_sugar" value={formData.fasting_blood_sugar} onChange={handleChange} className="glass-input w-full p-3 bg-white/40" required>
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 ml-1">Resting ECG</label>
                <select name="resting_ecg" value={formData.resting_ecg} onChange={handleChange} className="glass-input w-full p-3 bg-white/40" required>
                  <option value="Normal">Normal</option>
                  <option value="ST-T Wave Abnormality">ST-T Wave Abnormality</option>
                  <option value="Left Ventricular Hypertrophy">Left Ventricular Hypertrophy</option>
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 ml-1">Exercise Angina</label>
                <select name="exercise_induced_angina" value={formData.exercise_induced_angina} onChange={handleChange} className="glass-input w-full p-3 bg-white/40" required>
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 ml-1">ST Depression</label>
                <input type="number" step="0.1" name="st_depression" value={formData.st_depression} onChange={handleChange} className="glass-input w-full p-3 bg-white/40" required />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 ml-1">ST Slope</label>
                <select name="st_slope" value={formData.st_slope} onChange={handleChange} className="glass-input w-full p-3 bg-white/40" required>
                  <option value="Upsloping">Upsloping</option>
                  <option value="Flat">Flat</option>
                  <option value="Downsloping">Downsloping</option>
                </select>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 ml-1">Major Vessels (0-3)</label>
                <input type="number" name="num_major_vessels" value={formData.num_major_vessels} onChange={handleChange} min="0" max="3" className="glass-input w-full p-3 bg-white/40" required />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 ml-1">Thalassemia</label>
                <select name="thalassemia" value={formData.thalassemia} onChange={handleChange} className="glass-input w-full p-3 bg-white/40" required>
                  <option value="Normal">Normal</option>
                  <option value="Fixed Defect">Fixed Defect</option>
                  <option value="Reversible Defect">Reversible Defect</option>
                </select>
              </div>
            </div>

            <button type="submit" disabled={loading} className="w-full mt-4 bg-slate-800 hover:bg-slate-900 text-white font-medium py-4 rounded-xl shadow-lg hover:shadow-xl transition-all disabled:opacity-70">
              {loading ? <span className="animate-pulse">Synthesizing...</span> : 'Run ML Inference'}
            </button>
          </form>
        </div>

        <div className="lg:col-span-5 h-full">
          <div className="glass-panel p-8 h-full flex flex-col">
            <h2 className="text-xl font-outfit font-semibold mb-6 text-slate-800">Inference Results</h2>
            {result ? (
              <div className="space-y-6 flex-grow flex flex-col animate-fade-in-up">
                <div className={`p-6 rounded-2xl border ${result.prediction === 1 ? 'bg-red-400/20 border-red-400/30 text-red-900' : 'bg-emerald-400/20 border-emerald-400/30 text-emerald-900'} backdrop-blur-md`}>
                  <div className="text-sm font-medium opacity-80 uppercase tracking-widest mb-1">Model Classification</div>
                  <div className="text-3xl font-outfit font-bold">
                    {result.prediction_label}
                  </div>
                  <div className="mt-4 flex items-center justify-between text-sm">
                    <span>Risk Probability:</span>
                    <span className="font-bold text-lg">{(result.risk_probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-white/40 h-2 rounded-full mt-2 overflow-hidden">
                    <div className={`h-full ${result.prediction === 1 ? 'bg-red-500' : 'bg-emerald-500'}`} style={{ width: `${result.risk_probability * 100}%` }}></div>
                  </div>
                </div>
                
                <div className="flex-grow">
                  <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wider mb-3 mt-4">SHAP Explainability</h3>
                  
                  <div className="space-y-4">
                    <div>
                      <div className="text-xs font-semibold text-red-600 mb-2">Risk Increasing Factors</div>
                      <div className="space-y-2">
                        {result.explanation.risk_increasing_factors.map((f: any, i: number) => (
                          <div key={i} className="flex justify-between items-center text-sm bg-white/40 p-2 rounded-lg border border-white/50">
                            <span className="text-slate-700">{f.label} <span className="opacity-50 text-xs ml-1">({f.value})</span></span>
                            <span className="font-mono font-medium text-red-600">+{f.contribution.toFixed(3)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <div className="text-xs font-semibold text-emerald-600 mb-2 mt-4">Risk Decreasing Factors</div>
                      <div className="space-y-2">
                        {result.explanation.risk_decreasing_factors.map((f: any, i: number) => (
                          <div key={i} className="flex justify-between items-center text-sm bg-white/40 p-2 rounded-lg border border-white/50">
                            <span className="text-slate-700">{f.label} <span className="opacity-50 text-xs ml-1">({f.value})</span></span>
                            <span className="font-mono font-medium text-emerald-600">{f.contribution.toFixed(3)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="text-xs text-slate-500 mt-auto pt-6 border-t border-slate-200/50">
                  <span className="opacity-70">Model Version: {result.version}</span>
                </div>
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-slate-400 opacity-60">
                <svg className="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>
                <p>Awaiting patient inputs</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
