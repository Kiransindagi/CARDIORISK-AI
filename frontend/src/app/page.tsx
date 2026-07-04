import Link from 'next/link';

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] max-w-4xl mx-auto text-center space-y-8 animate-fade-in-up">
      <div className="inline-block px-4 py-1.5 rounded-full glass-panel text-sm font-semibold text-emerald-700 mb-4 tracking-wide shadow-sm">
        Next-Gen Predictive Analytics
      </div>
      <h1 className="text-6xl font-outfit font-extrabold tracking-tight text-slate-800 leading-tight">
        Explainable Intelligence for <br />
        <span className="bg-gradient-to-r from-emerald-500 to-teal-500 bg-clip-text text-transparent drop-shadow-sm">Heart Disease Risk</span>
      </h1>
      <p className="text-xl text-slate-600 max-w-2xl leading-relaxed font-light">
        A minimalist, industry-grade machine learning platform leveraging strict probability calibration, optimized thresholding, and liquid glass aesthetics.
      </p>
      
      <div className="flex flex-col sm:flex-row gap-6 mt-8 w-full justify-center">
        <Link href="/assessment" className="group relative overflow-hidden bg-gradient-to-r from-emerald-500 to-teal-500 text-white font-medium py-4 px-10 rounded-2xl shadow-lg hover:shadow-xl hover:-translate-y-1 transition-all duration-300">
          <span className="relative z-10 flex items-center gap-2">Launch Assessment <span className="group-hover:translate-x-1 transition-transform">&rarr;</span></span>
        </Link>
        <Link href="/performance" className="glass-panel hover:bg-white/40 text-slate-700 font-medium py-4 px-10 hover:-translate-y-1 transition-all duration-300 shadow-md">
          Explore Metrics
        </Link>
      </div>
    </div>
  );
}
