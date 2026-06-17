/* finder/static/finder/js/main.js */

'use strict';

// ── Navbar scroll effect ──────────────────────────────────────────────────
const mainNav = document.getElementById('mainNav');
if (mainNav) {
  window.addEventListener('scroll', () => {
    if (window.scrollY > 30) {
      mainNav.style.boxShadow = '0 4px 30px rgba(0,0,0,0.4)';
    } else {
      mainNav.style.boxShadow = '0 2px 20px rgba(0,0,0,0.3)';
    }
  }, { passive: true });
}

// ── Animated counter for stat numbers ─────────────────────────────────────
function animateCounters() {
  document.querySelectorAll('.stat-num[data-count]').forEach(el => {
    const target = parseFloat(el.dataset.count);
    const suffix = el.dataset.suffix || '';
    const duration = 1200;
    const start = performance.now();

    function step(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      el.textContent = Math.round(target * eased) + suffix;
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  });
}

// Trigger counter when stat cards enter viewport
const statsObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      animateCounters();
      statsObserver.disconnect();
    }
  });
}, { threshold: 0.3 });

const statsSection = document.querySelector('.stat-card');
if (statsSection) statsObserver.observe(statsSection);

// ── Scheme card entrance animation (scroll-triggered) ────────────────────
const cardObserver = new IntersectionObserver(entries => {
  entries.forEach((entry, i) => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
      cardObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

document.querySelectorAll('.scheme-card-animate').forEach((card, i) => {
  card.style.opacity = '0';
  card.style.transform = 'translateY(24px)';
  card.style.transition = `opacity 0.5s ease ${i * 0.1}s, transform 0.5s ease ${i * 0.1}s`;
  cardObserver.observe(card);
});

// ── Tooltip init (Bootstrap) ──────────────────────────────────────────────
document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
  new bootstrap.Tooltip(el);
});

// ── Form validation visual feedback ──────────────────────────────────────
const eligForm = document.getElementById('eligibilityForm');
if (eligForm) {
  eligForm.querySelectorAll('select, input').forEach(el => {
    el.addEventListener('change', () => {
      if (el.value && el.value !== '') {
        el.classList.add('is-valid');
        el.classList.remove('is-invalid');
      }
    });
  });
}

// ── Copy scheme name to clipboard ────────────────────────────────────────
document.querySelectorAll('.scheme-card h3').forEach(h => {
  h.style.cursor = 'pointer';
  h.title = 'Click to copy scheme name';
  h.addEventListener('click', () => {
    navigator.clipboard.writeText(h.textContent.trim()).then(() => {
      const toast = document.createElement('div');
      toast.textContent = '✓ Copied!';
      toast.className = 'copy-toast';
      toast.style.cssText =
        'position:fixed;bottom:24px;right:24px;background:#15803D;color:#fff;' +
        'padding:8px 18px;border-radius:999px;font-size:.85rem;z-index:9999;' +
        'animation:fadeInOut 1.8s ease forwards;';
      document.body.appendChild(toast);
      setTimeout(() => toast.remove(), 1800);
    }).catch(() => {});
  });
});

// Inject keyframes for toast
const toastStyle = document.createElement('style');
toastStyle.textContent =
  '@keyframes fadeInOut{0%{opacity:0;transform:translateY(10px)}' +
  '20%{opacity:1;transform:translateY(0)}' +
  '80%{opacity:1}100%{opacity:0}}';
document.head.appendChild(toastStyle);

