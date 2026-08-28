"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useState } from "react";

const races = [
  { name: "MONACO", date: "07 JUN", country: "MC", status: "NEXT" },
  { name: "SILVERSTONE", date: "05 JUL", country: "UK", status: "OPEN" },
  { name: "SPA", date: "19 JUL", country: "BE", status: "OPEN" },
  { name: "MONZA", date: "06 SEP", country: "IT", status: "OPEN" },
];

const features = [
  ["UPCOMING EVENTS", "Discover future races."],
  ["AI PREDICTIONS", "Explore predicted race results."],
  ["RACE ANALYTICS", "Read driver and circuit signals."],
  ["HISTORICAL DATA", "Compare previous race outcomes."],
];

const steps = [
  ["01", "CHOOSE A RACE"],
  ["02", "ANALYZE THE DATA"],
  ["03", "GENERATE PREDICTION"],
  ["04", "EXPLORE RESULTS"],
];

export function LandingPage() {
  const [scrollProgress, setScrollProgress] = useState(0);

  useEffect(() => {
    let frame = 0;
    const update = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const progress = Math.min(window.scrollY / Math.max(window.innerHeight * 0.9, 1), 1);
        setScrollProgress(progress);
      });
    };

    update();
    window.addEventListener("scroll", update, { passive: true });
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("scroll", update);
    };
  }, []);

  return (
    <div
      className="landing"
      style={
        {
          "--hero-progress": scrollProgress,
        } as React.CSSProperties
      }
    >
      <section className="hero-section">
        <Image
          alt="Generic motorsport car racing at sunset"
          className="hero-image"
          fill
          priority
          sizes="100vw"
          src="/mockup/hero-image.png"
        />
        <div className="hero-vignette" />
        <div className="hero-fade" />
        <div className="hero-grid" />

        <div className="hero-content">
          <p className="tech-label">ML RACE INTELLIGENCE</p>
          <h1>
            PREDICT.
            <br />
            ANALYZE.
            <br />
            <span>WIN.</span>
          </h1>
          <p className="hero-copy">AI-powered motorsport predictions.</p>
          <Link className="red-cta" href="/predicts">
            Explore Predictions <span>→</span>
          </Link>
        </div>

        <div className="hero-telemetry" aria-hidden="true">
          <span>SECTOR 03</span>
          <span>RACE PACE</span>
          <span>MODEL READY</span>
        </div>

        <div className="scroll-cue" aria-hidden="true">
          <span />
          <small>SCROLL</small>
        </div>
      </section>

      <section className="dark-entry" id="about">
        <p className="tech-label reveal">BUILT FOR RACING DATA</p>
        <h2 className="reveal">Predict what happens next.</h2>
      </section>

      <section className="landing-section upcoming-section">
        <div className="section-heading reveal">
          <p className="tech-label">UPCOMING RACES</p>
          <h2>Select the next event.</h2>
          <Link href="/races">View all races →</Link>
        </div>
        <div className="race-strip">
          {races.map((race, index) => (
            <article className={`event-tile reveal ${index === 0 ? "event-tile-active" : ""}`} key={race.name}>
              <div>
                <span>{race.status}</span>
                <strong>{race.name}</strong>
              </div>
              <svg viewBox="0 0 180 86" role="img" aria-label={`${race.name} circuit outline`}>
                <path d="M19 53 C48 12, 93 7, 124 28 C156 50, 159 73, 121 74 C88 75, 79 52, 48 66 C32 73, 14 70, 19 53Z" />
              </svg>
              <footer>
                <b>{race.date}</b>
                <span>{race.country}</span>
              </footer>
            </article>
          ))}
        </div>
      </section>

      <section className="landing-section prediction-experience">
        <div className="prediction-copy reveal">
          <p className="tech-label">REAL PREDICTIONS</p>
          <h2>
            SEE THE FUTURE
            <br />
            BEFORE THE LIGHTS GO OUT.
          </h2>
          <p>Race predictions powered by historical performance data.</p>
          <Link className="red-cta" href="/predicts">
            View Predictions <span>→</span>
          </Link>
        </div>

        <div className="dashboard-mock reveal">
          <aside>
            <b>Predict Race</b>
            <span>Dashboard</span>
            <span>Races</span>
            <span>Predictions</span>
            <span>History</span>
          </aside>
          <main>
            <div className="mock-card mock-wide">
              <small>UPCOMING RACE</small>
              <h3>Monaco Grand Prix</h3>
              <p>Race date · Circuit confidence · Dry conditions</p>
              <svg viewBox="0 0 200 90">
                <path d="M22 55 C35 22, 77 21, 94 43 S130 65, 151 33 C160 18, 183 28, 171 48 C158 70, 121 78, 86 64 C59 53, 43 77, 22 55Z" />
              </svg>
            </div>
            <div className="mock-card prediction-list">
              <small>TOP PREDICTION</small>
              {["Driver A", "Driver B", "Driver C", "Driver D", "Driver E"].map((driver, index) => (
                <div key={driver}>
                  <span>{index + 1}. {driver}</span>
                  <i style={{ width: `${72 - index * 11}%` }} />
                  <b>{[31, 23, 18, 15, 13][index]}%</b>
                </div>
              ))}
            </div>
          </main>
          <section>
            <div className="probability-ring">31%</div>
            <div className="mock-card compact">
              <small>TRACK STATS</small>
              <p>Turns 19</p>
              <p>Length 3.337 km</p>
              <p>Lap pace 1:12</p>
            </div>
          </section>
        </div>
      </section>

      <section className="photo-break">
        <Image
          alt="Generic racing circuit at high speed"
          className="photo-break-image"
          fill
          sizes="100vw"
          src="/mockup/hero-image.png"
        />
        <div className="photo-break-fade" />
        <h2 className="reveal">
          DATA DOESN&apos;T GUESS.
          <br />
          IT <span>LEARNS.</span>
        </h2>
      </section>

      <section className="landing-section feature-line">
        {features.map(([title, copy]) => (
          <article className="feature-item reveal" key={title}>
            <span />
            <h3>{title}</h3>
            <p>{copy}</p>
          </article>
        ))}
      </section>

      <section className="landing-section process-section" id="how-it-works">
        <p className="tech-label reveal">HOW IT WORKS</p>
        <div className="process-line">
          {steps.map(([number, title]) => (
            <article className="process-step reveal" key={number}>
              <span>{number}</span>
              <h3>{title}</h3>
            </article>
          ))}
        </div>
      </section>

      <section className="final-cta">
        <div className="cta-glow" />
        <h2 className="reveal">READY FOR THE NEXT RACE?</h2>
        <Link className="red-cta reveal" href="/predicts">
          Explore Predictions <span>→</span>
        </Link>
      </section>

      <footer className="landing-footer">
        <Link href="/" className="brand-mark">
          <span>Predict</span>
          <span>Race</span>
        </Link>
        <nav>
          <Link href="/predicts">Predictions</Link>
          <Link href="/races">Races</Link>
          <Link href="/#about">About</Link>
        </nav>
        <p>© 2026 F1 ML Predicts</p>
      </footer>
    </div>
  );
}
