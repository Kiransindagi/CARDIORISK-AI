import type { Metadata } from "next";
import { Inter, Outfit } from "next/font/google";
import "./globals.css";
import Link from "next/link";

const inter = Inter({ subsets: ["latin"], variable: '--font-inter' });
const outfit = Outfit({ subsets: ["latin"], variable: '--font-outfit' });

export const metadata: Metadata = {
  title: "CardioRisk AI",
  description: "Explainable Heart Disease Risk Prediction System",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} ${outfit.variable} font-sans min-h-screen flex flex-col text-slate-800`}>
        <div className="fixed inset-0 z-[-1] bg-gradient-to-br from-indigo-100 via-purple-50 to-teal-100"></div>
        <div className="fixed top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-blue-300/30 blur-3xl z-[-1]"></div>
        <div className="fixed bottom-[-20%] right-[-10%] w-[60%] h-[60%] rounded-full bg-teal-300/20 blur-3xl z-[-1]"></div>

        <header className="sticky top-0 z-50 mt-4 mx-4 md:mx-auto max-w-5xl glass-panel px-6 py-4 flex items-center justify-between transition-all duration-300">
          <div className="flex-shrink-0">
            <Link href="/" className="font-outfit font-bold text-2xl tracking-tight text-slate-800 flex items-center gap-2">
              <span className="bg-gradient-to-r from-emerald-500 to-teal-600 bg-clip-text text-transparent">CardioRisk</span> AI
            </Link>
          </div>
          <nav className="hidden md:flex items-center space-x-1">
            <Link href="/" className="hover:bg-white/30 px-4 py-2 rounded-full text-sm font-medium transition-colors">Overview</Link>
            <Link href="/assessment" className="hover:bg-white/30 px-4 py-2 rounded-full text-sm font-medium transition-colors">Assessment</Link>
            <Link href="/performance" className="hover:bg-white/30 px-4 py-2 rounded-full text-sm font-medium transition-colors">Performance</Link>
            <Link href="/explainability" className="hover:bg-white/30 px-4 py-2 rounded-full text-sm font-medium transition-colors">Explainability</Link>
            <Link href="/model-card" className="hover:bg-white/30 px-4 py-2 rounded-full text-sm font-medium transition-colors">Model Card</Link>
          </nav>
        </header>
        <main className="flex-grow w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12 flex flex-col justify-center">
          {children}
        </main>
        <footer className="py-8 text-center text-sm text-slate-500 font-medium">
          <p>CardioRisk AI is an educational demonstration. Not for medical diagnosis.</p>
        </footer>
      </body>
    </html>
  );
}
