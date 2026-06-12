/**
 * Focus Labs — interactions
 * Preloader, split-line headline reveals, scroll reveals, reactive
 * "imagine" lines, scroll progress, parallax, counters, magnetic buttons,
 * card tilt, custom cursor, nav state, early-access form, replay demo.
 */
(function () {
  "use strict";

  var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var hasIO = "IntersectionObserver" in window;
  var finePointer = window.matchMedia("(pointer: fine)").matches;

  /* ---------------- Preloader ---------------- */
  /* All entrance animations start only after the preloader lifts,
     so nothing plays behind the curtain. */

  var preloader = document.getElementById("preloader");
  var countEl = document.getElementById("preloaderCount");
  var barEl = document.getElementById("preloaderBar");
  var motionStarted = false;

  function finishPreloader() {
    if (preloader) preloader.classList.add("is-done");
    document.body.classList.add("is-loaded");
    if (!motionStarted) {
      motionStarted = true;
      initMotion();
    }
  }

  if (preloader && !reducedMotion) {
    var progress = 0;
    var timer = setInterval(function () {
      progress = Math.min(progress + 6 + Math.random() * 14, 100);
      var v = Math.floor(progress);
      if (countEl) countEl.textContent = v;
      if (barEl) barEl.style.width = v + "%";
      if (progress >= 100) {
        clearInterval(timer);
        setTimeout(finishPreloader, 350);
      }
    }, 90);
    // Safety net: never trap the user behind the preloader
    setTimeout(finishPreloader, 4000);
  } else {
    finishPreloader();
  }

  /* ---------------- Entrance motion (post-preloader) ---------------- */

  function initMotion() {
    var revealEls = document.querySelectorAll("[data-reveal]");
    revealEls.forEach(function (el) {
      var delay = el.getAttribute("data-reveal-delay");
      if (delay) el.style.setProperty("--reveal-delay", delay + "ms");
    });

    var splitEls = document.querySelectorAll("[data-split]");

    if (!hasIO || reducedMotion) {
      revealEls.forEach(function (el) { el.classList.add("is-visible"); });
      splitEls.forEach(function (el) { el.classList.add("is-inview"); });
      document.querySelectorAll("[data-counter]").forEach(function (el) {
        el.textContent = el.getAttribute("data-counter");
      });
      document.querySelectorAll(".imagine__line").forEach(function (el) {
        el.classList.add("is-active");
      });
      return;
    }

    /* Fade/rise reveals */
    var revealObserver = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            revealObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15, rootMargin: "0px 0px -40px 0px" }
    );
    revealEls.forEach(function (el) { revealObserver.observe(el); });

    /* Split-line headline reveals: each .line rises out of a mask */
    splitEls.forEach(splitLines);
    var splitObserver = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-inview");
            splitObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.3 }
    );
    splitEls.forEach(function (el) { splitObserver.observe(el); });

    /* Counters */
    var counterObserver = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            animateCounter(entry.target);
            counterObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.6 }
    );
    document.querySelectorAll("[data-counter]").forEach(function (el) {
      counterObserver.observe(el);
    });

    /* "Imagine" lines wake up while they cross the middle of the
       viewport and dim again when they leave — both directions. */
    var imagineObserver = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          entry.target.classList.toggle("is-active", entry.isIntersecting);
        });
      },
      { rootMargin: "-32% 0px -32% 0px", threshold: 0 }
    );
    document.querySelectorAll(".imagine__line").forEach(function (el) {
      imagineObserver.observe(el);
    });
  }

  function splitLines(el) {
    var lines = el.querySelectorAll(".line");
    if (!lines.length) lines = [el];
    Array.prototype.forEach.call(lines, function (line, i) {
      var inner = document.createElement("span");
      inner.className = "sli";
      inner.style.setProperty("--i", i);
      while (line.firstChild) inner.appendChild(line.firstChild);
      line.appendChild(inner);
      line.classList.add("sl");
    });
  }

  /* ---------------- Animated counters ---------------- */

  function animateCounter(el) {
    var target = parseFloat(el.getAttribute("data-counter"));
    var decimals = parseInt(el.getAttribute("data-counter-decimal") || "0", 10);
    var compact = el.getAttribute("data-counter-format") === "compact";
    var duration = 1800;
    var start = null;

    function format(value) {
      if (decimals) return (value / Math.pow(10, decimals)).toFixed(decimals);
      if (compact) {
        if (value >= 1e6) return (value / 1e6).toFixed(1).replace(/\.0$/, "") + "M";
        if (value >= 1e3) return Math.round(value / 1e3) + "K";
      }
      return Math.round(value).toLocaleString();
    }

    function step(ts) {
      if (!start) start = ts;
      var p = Math.min((ts - start) / duration, 1);
      var eased = 1 - Math.pow(1 - p, 4);
      el.textContent = format(target * eased);
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  /* ---------------- Nav + scroll progress ---------------- */

  var nav = document.getElementById("nav");
  var progressBar = document.getElementById("scrollProgress");

  var onScroll = function () {
    if (nav) nav.classList.toggle("is-scrolled", window.scrollY > 24);
    if (progressBar) {
      var max = document.documentElement.scrollHeight - window.innerHeight;
      progressBar.style.transform = "scaleX(" + (max > 0 ? window.scrollY / max : 0) + ")";
    }
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll, { passive: true });
  onScroll();

  var burger = document.getElementById("navBurger");
  var navLinks = document.getElementById("navLinks");
  if (burger && navLinks) {
    burger.addEventListener("click", function () {
      var open = navLinks.classList.toggle("is-open");
      burger.setAttribute("aria-expanded", String(open));
      document.body.style.overflow = open ? "hidden" : "";
    });
    navLinks.querySelectorAll("a").forEach(function (a) {
      a.addEventListener("click", function () {
        navLinks.classList.remove("is-open");
        burger.setAttribute("aria-expanded", "false");
        document.body.style.overflow = "";
      });
    });
  }

  /* ---------------- Parallax drift ---------------- */

  var parallaxEls = [];
  document.querySelectorAll("[data-parallax]").forEach(function (el) {
    parallaxEls.push({ el: el, speed: parseFloat(el.getAttribute("data-parallax")) || 0.08 });
  });

  if (parallaxEls.length && !reducedMotion) {
    var parallaxTicking = false;

    var updateParallax = function () {
      var vh = window.innerHeight;
      parallaxEls.forEach(function (p) {
        var r = p.el.getBoundingClientRect();
        if (r.bottom < -100 || r.top > vh + 100) return;
        var offset = (r.top + r.height / 2 - vh / 2) * p.speed;
        p.el.style.transform = "translate3d(0," + -offset.toFixed(1) + "px,0)";
      });
      parallaxTicking = false;
    };

    window.addEventListener(
      "scroll",
      function () {
        if (!parallaxTicking) {
          parallaxTicking = true;
          requestAnimationFrame(updateParallax);
        }
      },
      { passive: true }
    );
    updateParallax();
  }

  /* ---------------- Magnetic buttons ---------------- */

  if (finePointer && !reducedMotion) {
    document.querySelectorAll("[data-magnetic]").forEach(function (el) {
      var strength = 0.25;
      el.addEventListener("mousemove", function (e) {
        var r = el.getBoundingClientRect();
        var x = e.clientX - r.left - r.width / 2;
        var y = e.clientY - r.top - r.height / 2;
        el.style.transform = "translate(" + x * strength + "px," + y * strength + "px)";
      });
      el.addEventListener("mouseleave", function () {
        el.style.transform = "";
      });
    });
  }

  /* ---------------- Card tilt ---------------- */

  if (finePointer && !reducedMotion) {
    document.querySelectorAll("[data-tilt]").forEach(function (el) {
      var max = parseFloat(el.getAttribute("data-tilt-max") || "4");
      el.style.transformStyle = "preserve-3d";
      el.addEventListener("mousemove", function (e) {
        var r = el.getBoundingClientRect();
        var px = (e.clientX - r.left) / r.width - 0.5;
        var py = (e.clientY - r.top) / r.height - 0.5;
        el.style.transform =
          "perspective(900px) rotateX(" + (-py * max) + "deg) rotateY(" + (px * max) + "deg) translateY(-2px)";
      });
      el.addEventListener("mouseleave", function () {
        el.style.transform = "";
      });
    });
  }

  /* ---------------- Custom cursor ---------------- */

  var cursor = document.getElementById("cursor");
  var ring = document.getElementById("cursorRing");

  if (cursor && ring && finePointer && !reducedMotion) {
    var cx = -100, cy = -100, rx = -100, ry = -100;

    document.addEventListener("mousemove", function (e) {
      cx = e.clientX;
      cy = e.clientY;
      document.body.classList.add("cursor-active");
    });

    document.addEventListener("mouseleave", function () {
      document.body.classList.remove("cursor-active");
    });

    var hoverables = "a, button, [data-tilt]";
    document.addEventListener("mouseover", function (e) {
      document.body.classList.toggle("cursor-hover", !!e.target.closest(hoverables));
    });

    (function loop() {
      rx += (cx - rx) * 0.16;
      ry += (cy - ry) * 0.16;
      cursor.style.transform = "translate(" + (cx - 3) + "px," + (cy - 3) + "px)";
      ring.style.transform = "translate(" + rx + "px," + ry + "px) translate(-50%,-50%)";
      requestAnimationFrame(loop);
    })();
  }

  /* ---------------- Device replay demo ---------------- */

  var timeEl = document.getElementById("deviceTime");
  var progressEl = document.getElementById("deviceProgress");

  if (timeEl && progressEl && !reducedMotion) {
    var TOTAL = 4 * 3600 + 12 * 60; // the "4h 12m" session, in seconds
    var REPLAY_MS = 9000;
    var startTs = null;

    var pad = function (n) { return String(n).padStart(2, "0"); };

    var tick = function (ts) {
      if (!startTs) startTs = ts;
      var p = ((ts - startTs) % (REPLAY_MS + 1200)) / REPLAY_MS;
      p = Math.min(p, 1); // brief hold at the end before looping
      var s = Math.floor(TOTAL * p);
      timeEl.textContent =
        pad(Math.floor(s / 3600)) + ":" + pad(Math.floor((s % 3600) / 60)) + ":" + pad(s % 60);
      progressEl.style.width = p * 100 + "%";
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  } else if (timeEl) {
    timeEl.textContent = "04:12:00";
    if (progressEl) progressEl.style.width = "100%";
  }

  /* ---------------- Sticky mobile CTA ---------------- */
  /* Appears once the hero is scrolled past, retreats while the real
     CTA section (or footer) is on screen. */

  var stickyCta = document.getElementById("stickyCta");
  var heroSection = document.getElementById("hero");
  var ctaSection = document.getElementById("cta");

  if (stickyCta && heroSection && ctaSection && hasIO) {
    var pastHero = false;
    var ctaOnScreen = false;

    var syncSticky = function () {
      stickyCta.classList.toggle("is-shown", pastHero && !ctaOnScreen);
    };

    new IntersectionObserver(function (entries) {
      pastHero = !entries[0].isIntersecting;
      syncSticky();
    }, { threshold: 0.05 }).observe(heroSection);

    new IntersectionObserver(function (entries) {
      ctaOnScreen = entries[0].isIntersecting;
      syncSticky();
    }, { threshold: 0.1 }).observe(ctaSection);
  }

  /* ---------------- FAQ accordion ---------------- */

  var faqItems = document.querySelectorAll(".faq__item");
  faqItems.forEach(function (item) {
    item.addEventListener("toggle", function () {
      if (!item.open) return;
      faqItems.forEach(function (other) {
        if (other !== item) other.open = false;
      });
    });
  });

  /* ---------------- Early access form ---------------- */

  var accessForm = document.getElementById("accessForm");
  var accessEmail = document.getElementById("accessEmail");
  var accessSuccess = document.getElementById("accessSuccess");

  if (accessForm && accessEmail && accessSuccess) {
    accessForm.addEventListener("submit", function (e) {
      e.preventDefault();
      var value = accessEmail.value.trim();
      var valid = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);

      if (!valid) {
        accessForm.classList.remove("is-error");
        // Force a reflow so the shake animation can replay
        void accessForm.offsetWidth;
        accessForm.classList.add("is-error");
        accessEmail.focus();
        return;
      }

      accessForm.hidden = true;
      accessSuccess.hidden = false;
    });

    accessEmail.addEventListener("input", function () {
      accessForm.classList.remove("is-error");
    });
  }

  /* ---------------- Footer year ---------------- */

  var yearEl = document.getElementById("year");
  if (yearEl) yearEl.textContent = new Date().getFullYear();
})();
