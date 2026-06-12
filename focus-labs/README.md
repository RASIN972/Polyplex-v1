# Focus Labs — Marketing Site

A premium single-page marketing site for **Focus Labs**, the app that turns
deep work sessions into cinematic timelapses.

The page follows a three-act narrative: **Act I** names the problem (your
effort is invisible, so it's easy to quit), **Act II** paints the ideal
future without naming the product ("imagine your whole workday, replayed in
forty seconds"), and **Act III** reveals Focus Labs with a waitlist form for
early access.

## Design

- **Palette** — silver chrome gradients on near-black (`#07080a`), with glass
  cards, film grain, and an animated silver-sheen headline treatment.
- **Hero** — a Three.js "time vortex": a 14,000-particle silver galaxy with a
  glowing core and orbit ring, with mouse parallax and a scroll-linked camera
  dolly.
- **Motion** — preloader with progress counter, scroll-triggered reveals,
  animated stat counters, magnetic buttons, 3D card tilt, custom cursor,
  marquee strip, and a live "session replay" device demo (4h 12m → 42s).

## Running locally

It's a static site — no build step. Serve the folder over HTTP (ES modules
require it):

```sh
cd focus-labs
python3 -m http.server 8000
# open http://localhost:8000
```

Three.js is loaded from the jsDelivr CDN via an import map. If WebGL or the
CDN is unavailable, the site degrades gracefully — every section works
without the hero scene.

## Accessibility & performance

- Respects `prefers-reduced-motion` (disables the scene, grain, reveals, and
  cursor effects).
- Custom cursor and tilt only activate on fine-pointer devices.
- The WebGL loop pauses while the hero is off-screen; pixel ratio is capped
  at 2.
- Fully responsive, with a full-screen mobile menu under 720px.

## Structure

```
focus-labs/
├── index.html      # all sections & content
├── css/style.css   # design system + animations
└── js/
    ├── scene.js    # Three.js hero (ES module)
    └── main.js     # interactions (preloader, reveals, counters, cursor…)
```
