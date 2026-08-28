import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "F1 ML Predicts",
  description: "Simulador de predicciones F1 con un modelo local de ML.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="es" className="h-full antialiased">
      <body className="min-h-full bg-background text-foreground">
        <header className="site-header">
          <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-5">
            <Link href="/" className="brand-mark">
              <span>Predict</span>
              <span>Race</span>
            </Link>
            <nav className="hidden items-center gap-8 text-sm md:flex">
              <Link className="nav-link" href="/predicts">
                Predictions
              </Link>
              <Link className="nav-link" href="/races">
                Races
              </Link>
              <Link className="nav-link" href="/#how-it-works">
                How it works
              </Link>
              <Link className="nav-link" href="/#about">
                About
              </Link>
            </nav>
            <Link className="header-cta" href="/predicts">
              Explore Predictions
            </Link>
          </div>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
