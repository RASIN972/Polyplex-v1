/**
 * Focus Labs — interactions
 * Preloader, scroll reveals, animated counters, magnetic buttons,
 * card tilt, custom cursor, nav state, and the device replay demo.
 */
(function () {
  "use strict";

  var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------- Preloader ---------------- */

  var preloader = document.getElementById("preloader");
  var countEl = document.getElementById("preloaderCount");
  var barEl = document.getElementById("preloaderBar");

  function finishPreloader() {
    if (preloader) preloader.classList.add("is-done");
    document.body.classList.add("is-loaded");
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

  /* ---------------- Scroll reveals ---------------- */

  var revealEls = document.querySelectorAll("[data-reveal]");
  revealEls.forEach(function (el) {
    var delay = el.getAttribute("data-reveal-delay");
    if (delay) el.style.setProperty("--reveal-delay", delay + "ms");
  });

  if ("IntersectionObserver" in window && !reducedMotion) {
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
  } else {
    revealEls.forEach(function (el) { el.classList.add("is-visible"); });
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

  var counters = document.querySelectorAll("[data-counter]");
  if ("IntersectionObserver" in window && !reducedMotion) {
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
    counters.forEach(function (el) { counterObserver.observe(el); });
  } else {
    counters.forEach(function (el) {
      el.textContent = el.getAttribute("data-counter");
      animateCounter(el);
    });
  }

  /* ---------------- Nav ---------------- */

  var nav = document.getElementById("nav");
  var onScroll = function () {
    if (nav) nav.classList.toggle("is-scrolled", window.scrollY > 24);
  };
  window.addEventListener("scroll", onScroll, { passive: true });
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

  /* ---------------- Magnetic buttons ---------------- */

  if (window.matchMedia("(pointer: fine)").matches && !reducedMotion) {
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

  if (window.matchMedia("(pointer: fine)").matches && !reducedMotion) {
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

  if (cursor && ring && window.matchMedia("(pointer: fine)").matches && !reducedMotion) {
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

    function pad(n) { return String(n).padStart(2, "0"); }

    function tick(ts) {
      if (!startTs) startTs = ts;
      var p = ((ts - startTs) % (REPLAY_MS + 1200)) / REPLAY_MS;
      p = Math.min(p, 1); // brief hold at the end before looping
      var s = Math.floor(TOTAL * p);
      timeEl.textContent =
        pad(Math.floor(s / 3600)) + ":" + pad(Math.floor((s % 3600) / 60)) + ":" + pad(s % 60);
      progressEl.style.width = p * 100 + "%";
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  } else if (timeEl) {
    timeEl.textContent = "04:12:00";
    if (progressEl) progressEl.style.width = "100%";
  }

  /* ---------------- Footer year ---------------- */

  var yearEl = document.getElementById("year");
  if (yearEl) yearEl.textContent = new Date().getFullYear();
})();
