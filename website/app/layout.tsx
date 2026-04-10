import type { Metadata } from "next";

/**
 * Site-wide metadata.
 *
 * SEO Fix 1 — Title / Content keyword alignment
 *   The page title includes the word "Investing".  The same word appears
 *   in the body copy on page.tsx so crawlers find exact-match keyword
 *   evidence in both the <title> and the page content.
 *
 * SEO Fix 2 — Favicon referenced in HTML
 *   Next.js 13+ App Router auto-discovers `app/favicon.ico`, but many
 *   crawlers and older browsers still expect an explicit <link rel="icon">
 *   tag.  The `icons` field below emits those tags into every page's
 *   <head> automatically.
 */
export const metadata: Metadata = {
  /* ── Basic identity ─────────────────────────────────────────────────── */
  title: {
    default: "Aphelion Research — Systematic Quantitative Investing",
    template: "%s | Aphelion Research",
  },
  description:
    "Aphelion Research is a systematic investment research firm specialising in " +
    "gold market microstructure, systematic macro, signal design, and regime " +
    "detection. Rigorous quantitative methods. No discretionary bias.",

  /* ── Canonical URL ───────────────────────────────────────────────────── */
  metadataBase: new URL("https://www.aphelionresearch.ca"),
  alternates: { canonical: "/" },

  /* ── SEO Fix 2: Favicon — all variants explicitly declared ──────────── */
  icons: {
    // Standard browser favicon  → <link rel="icon" href="/favicon.ico" …>
    icon: [
      { url: "/favicon.ico", sizes: "any" },
      { url: "/favicon.svg", type: "image/svg+xml" },
      { url: "/favicon-96x96.png", type: "image/png", sizes: "96x96" },
    ],
    // Safari pinned tab
    shortcut: "/favicon.ico",
    // iOS home-screen icon  → <link rel="apple-touch-icon" href="…">
    apple: [{ url: "/apple-touch-icon.png", sizes: "180x180" }],
  },

  /* ── Open Graph ─────────────────────────────────────────────────────── */
  openGraph: {
    title: "Aphelion Research — Systematic Quantitative Investing",
    description:
      "Systematic investment research: gold microstructure, macro regimes, " +
      "and signal design.",
    url: "https://www.aphelionresearch.ca",
    siteName: "Aphelion Research",
    locale: "en_CA",
    type: "website",
  },

  /* ── Twitter / X card ────────────────────────────────────────────────── */
  twitter: {
    card: "summary",
    title: "Aphelion Research — Systematic Quantitative Investing",
    description:
      "Systematic investment research: gold microstructure, macro regimes, " +
      "and signal design.",
  },

  /* ── Robots ─────────────────────────────────────────────────────────── */
  robots: {
    index: true,
    follow: true,
    googleBot: { index: true, follow: true },
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    /*
     * Next.js App Router injects all <link rel="icon"> and <meta> tags
     * declared in the `metadata` export above directly into the server-
     * rendered <head>, so no manual <link> tags are needed here.
     */
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
