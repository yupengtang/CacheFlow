(function () {
  // ── Scroll spy ──────────────────────────
  var sections = document.querySelectorAll('section[id]');
  var navLinks = document.querySelectorAll('.nav-links a');

  function onScroll() {
    var y = window.scrollY + 90;
    var current = '';
    sections.forEach(function (s) {
      if (s.offsetTop <= y) current = s.id;
    });
    navLinks.forEach(function (a) {
      a.classList.toggle('active', a.getAttribute('href') === '#' + current);
    });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  // ── Reveal on scroll ────────────────────
  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        observer.unobserve(e.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -30px 0px' });

  document.querySelectorAll('.card, .approach-item, .chart-box, .code-block')
    .forEach(function (el) {
      el.classList.add('reveal');
      observer.observe(el);
    });

  // ── Theme toggle ────────────────────────
  var root = document.documentElement;
  var btn = document.getElementById('theme-toggle');

  function getPreferred() {
    var stored = localStorage.getItem('cf-theme');
    if (stored) return stored;
    return window.matchMedia('(prefers-color-scheme: light)').matches
      ? 'light' : 'dark';
  }

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    localStorage.setItem('cf-theme', theme);

    if (typeof window.rebuildCharts === 'function') {
      window.rebuildCharts(theme);
    }
  }

  applyTheme(getPreferred());

  if (btn) {
    btn.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      applyTheme(next);
    });
  }

  window.matchMedia('(prefers-color-scheme: light)')
    .addEventListener('change', function (e) {
      if (!localStorage.getItem('cf-theme')) {
        applyTheme(e.matches ? 'light' : 'dark');
      }
    });
})();
