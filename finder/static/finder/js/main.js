/* finder/static/finder/js/main.js */

'use strict';

// ── Navbar scroll shadow effect ───────────────────────────────────────────
const mainNav = document.getElementById('mainNav');
if (mainNav) {
  window.addEventListener('scroll', () => {
    mainNav.style.boxShadow = window.scrollY > 30
      ? '0 4px 30px rgba(0,0,0,0.4)'
      : '0 2px 20px rgba(0,0,0,0.3)';
  }, { passive: true });
}

// ── Scheme card entrance animation (scroll-triggered) ─────────────────────
// Only runs on the results page where .scheme-card-animate elements exist.
const cards = document.querySelectorAll('.scheme-card-animate');
if (cards.length) {
  const cardObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
        cardObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

  cards.forEach((card, i) => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(24px)';
    card.style.transition = `opacity 0.5s ease ${i * 0.1}s, transform 0.5s ease ${i * 0.1}s`;
    cardObserver.observe(card);
  });

  // ── Copy scheme name to clipboard (click on card title) ────────────────
  // Keyframe injection is scoped here — only runs when scheme cards exist.
  const toastStyle = document.createElement('style');
  toastStyle.textContent =
    '@keyframes fadeInOut{0%{opacity:0;transform:translateY(10px)}' +
    '20%{opacity:1;transform:translateY(0)}80%{opacity:1}100%{opacity:0}}';
  document.head.appendChild(toastStyle);

  document.querySelectorAll('.scheme-card h3').forEach((h) => {
    h.style.cursor = 'pointer';
    h.title = 'Click to copy scheme name';
    h.addEventListener('click', () => {
      navigator.clipboard.writeText(h.textContent.trim()).then(() => {
        const toast = document.createElement('div');
        toast.textContent = '✓ Copied!';
        toast.style.cssText =
          'position:fixed;bottom:24px;right:24px;background:#15803D;color:#fff;' +
          'padding:8px 18px;border-radius:999px;font-size:.85rem;z-index:9999;' +
          'animation:fadeInOut 1.8s ease forwards;';
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 1800);
      }).catch(() => {});
    });
  });
}

// ── Form validation visual feedback ──────────────────────────────────────
// Only runs on the find/eligibility form page.
const eligForm = document.getElementById('eligibilityForm');
if (eligForm) {
  eligForm.querySelectorAll('select, input').forEach((el) => {
    el.addEventListener('change', () => {
      if (el.value && el.value !== '') {
        el.classList.add('is-valid');
        el.classList.remove('is-invalid');
      }
    });
  });
}
