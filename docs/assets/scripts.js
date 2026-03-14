(() => {
  const wrapCode = () => {
    document.querySelectorAll('pre > code').forEach((code) => {
      const pre = code.parentElement;
      if (pre.dataset.copyBound) return;
      pre.dataset.copyBound = '1';
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'copy-btn';
      btn.textContent = 'Copy';
      btn.addEventListener('click', async () => {
        const text = code.textContent || '';
        try {
          await navigator.clipboard.writeText(text);
          btn.textContent = 'Copied';
          setTimeout(() => (btn.textContent = 'Copy'), 900);
        } catch (_e) {
          btn.textContent = 'Unavailable';
          setTimeout(() => (btn.textContent = 'Copy'), 900);
        }
      });
      pre.appendChild(btn);
    });
  };

  const markActive = () => {
    const current = location.pathname.split('/').pop() || 'index.html';
    document.querySelectorAll('.nav-list a').forEach((link) => {
      const href = link.getAttribute('href') || '';
      if (href.endsWith(current)) {
        link.setAttribute('aria-current', 'page');
      }
    });
  };

  const init = () => {
    wrapCode();
    markActive();

    document.querySelectorAll('.api-section summary').forEach((el) => {
      el.addEventListener('click', () => {
        const sec = el.parentElement;
        const open = sec.hasAttribute('open');
        sec.setAttribute('aria-expanded', open ? 'false' : 'true');
      });
    });
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
