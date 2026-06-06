(function () {
  var sections = document.querySelectorAll('section[id]');
  var navLinks = document.querySelectorAll('.nav-links a');

  function onScroll() {
    var scrollY = window.scrollY + 80;
    var current = '';
    sections.forEach(function (s) {
      if (s.offsetTop <= scrollY) current = s.id;
    });
    navLinks.forEach(function (a) {
      a.classList.toggle('active', a.getAttribute('href') === '#' + current);
    });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  // subtle entrance for cards and approach items
  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        observer.unobserve(e.target);
      }
    });
  }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

  document.querySelectorAll('.card, .approach-item, .chart-box').forEach(function (el) {
    el.classList.add('reveal');
    observer.observe(el);
  });
})();
