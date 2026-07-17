(() => {
  document.documentElement.classList.add('js');
  const body = document.body;
  const sidebar = document.querySelector('.sidebar');
  const content = document.querySelector('.content');

  if (sidebar) {
    sidebar.id = 'site-navigation';
    const mobile = window.matchMedia('(max-width: 980px)');

    const menu = document.createElement('button');
    menu.type = 'button';
    menu.className = 'menu-button';
    menu.textContent = 'Menu';
    menu.setAttribute('aria-controls', sidebar.id);
    menu.setAttribute('aria-expanded', 'false');

    const scrim = document.createElement('button');
    scrim.type = 'button';
    scrim.className = 'nav-scrim';
    scrim.setAttribute('aria-label', 'Close navigation');

    const setOpen = (open) => {
      const hidden = mobile.matches && !open;
      body.classList.toggle('nav-open', open);
      menu.textContent = open ? 'Close' : 'Menu';
      menu.setAttribute('aria-expanded', String(open));
      sidebar.toggleAttribute('inert', hidden);
      content?.toggleAttribute('inert', mobile.matches && open);
      if (hidden) {
        sidebar.setAttribute('aria-hidden', 'true');
      } else {
        sidebar.removeAttribute('aria-hidden');
      }
    };

    menu.addEventListener('click', () => {
      const open = !body.classList.contains('nav-open');
      setOpen(open);
      if (open) {
        window.requestAnimationFrame(() => sidebar.querySelector('a')?.focus());
      }
    });
    scrim.addEventListener('click', () => {
      setOpen(false);
      menu.focus();
    });
    sidebar.addEventListener('click', (event) => {
      if (event.target.closest('a') && window.matchMedia('(max-width: 980px)').matches) {
        setOpen(false);
      }
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && body.classList.contains('nav-open')) {
        setOpen(false);
        menu.focus();
      }
    });

    body.append(menu, scrim);
    mobile.addEventListener('change', () => setOpen(false));
    setOpen(false);
  }

  const current = location.pathname.replace(/\/$/, '/index.html');
  const currentLinks = Array.from(document.querySelectorAll('.nav-list a')).filter((link) => {
    const path = new URL(link.href, location.href).pathname.replace(/\/$/, '/index.html');
    return path === current;
  });
  const currentLink = currentLinks[currentLinks.length - 1];
  currentLink?.classList.add('active');
  currentLink?.setAttribute('aria-current', 'page');

  document.querySelectorAll('pre > code').forEach((code) => {
    const pre = code.parentElement;
    if (pre.querySelector('.copy-btn')) return;

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy-btn';
    button.textContent = 'Copy';
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code.textContent || '');
        button.textContent = 'Copied';
      } catch (_error) {
        button.textContent = 'Select text';
      }
      window.setTimeout(() => (button.textContent = 'Copy'), 1200);
    });
    pre.appendChild(button);
  });

  document.querySelectorAll('[data-year]').forEach((node) => {
    node.textContent = new Date().getFullYear();
  });
})();
