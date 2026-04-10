/**
 * Aphelion Research — Home Page
 *
 * SEO Fix 1 — Title / Content keyword alignment
 *   The page <title> contains "Investing".  To satisfy the keyword-match
 *   requirement the word "investing" appears naturally in:
 *     1. The sub-headline beneath the H1 (primary placement)
 *     2. The "What we do" paragraph (secondary placement)
 *   Both placements are editorially appropriate for a systematic
 *   *investment* research firm and require no artificial stuffing.
 */
export default function HomePage() {
  return (
    <main>
      {/* ── Hero ──────────────────────────────────────────────────────── */}
      <section aria-labelledby="hero-heading">
        {/* H1 — kept exactly as designed; title keyword appears below */}
        <h1 id="hero-heading">
          Systematic research at the frontier of quantitative finance.
        </h1>

        {/*
          SEO Fix 1 — primary keyword placement.
          "investing" echoes the page <title> ("…Quantitative Investing")
          so crawlers confirm the keyword in both title and body.
        */}
        <p>
          Aphelion Research delivers institutional-grade systematic investing
          intelligence — built on rigorous quantitative methods, not discretionary
          intuition.
        </p>

        <a href="/research">Explore Our Research</a>
      </section>

      {/* ── What we do ────────────────────────────────────────────────── */}
      <section aria-labelledby="what-we-do-heading">
        <h2 id="what-we-do-heading">What We Do</h2>

        {/*
          SEO Fix 1 — secondary keyword placement.
          "investing" appears a second time in a contextually relevant
          sentence, reinforcing the title keyword without over-optimising.
        */}
        <p>
          We design and validate systematic investing strategies across four
          core domains:
        </p>

        <ul>
          <li>
            <strong>Gold Market Microstructure</strong> — order-flow toxicity,
            VPIN, intraday regime detection, and spread dynamics on XAUUSD.
          </li>
          <li>
            <strong>Systematic Macro</strong> — cross-asset regime models
            linking DXY, rates, and commodities to gold positioning.
          </li>
          <li>
            <strong>Signal Design</strong> — multi-horizon directional signals
            built on feature-engineered microstructure and macro inputs.
          </li>
          <li>
            <strong>Regime Detection</strong> — hidden Markov models and
            ensemble classifiers for real-time market-state identification.
          </li>
        </ul>
      </section>

      {/* ── Approach ──────────────────────────────────────────────────── */}
      <section aria-labelledby="approach-heading">
        <h2 id="approach-heading">Our Approach</h2>
        <p>
          Every insight at Aphelion Research is derived from data.  We combine
          high-frequency microstructure features, multi-timeframe technical
          signals, and macro co-integration factors inside a single ensemble
          model — trained end-to-end on live XAUUSD bar data.
        </p>
        <p>
          Research outputs are back-tested under realistic transaction-cost
          assumptions and validated in paper-trading before any live
          deployment.
        </p>
      </section>

      {/* ── Contact / CTA ─────────────────────────────────────────────── */}
      <section aria-labelledby="contact-heading">
        <h2 id="contact-heading">Get in Touch</h2>
        <p>
          Interested in systematic quantitative research or institutional
          collaboration?{" "}
          <a href="mailto:contact@aphelionresearch.ca">
            contact@aphelionresearch.ca
          </a>
        </p>
      </section>
    </main>
  );
}
