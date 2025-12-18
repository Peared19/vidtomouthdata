export function setStatus(message, opts = {}) {
  const el = document.getElementById('status');
  if (!el) return;

  el.textContent = String(message);
  el.classList.remove('hidden');

  if (opts.isError) {
    el.style.color = '#ffd1d1';
  } else {
    el.style.color = '#cfd8ff';
  }
}

export function clearStatus() {
  const el = document.getElementById('status');
  if (!el) return;
  el.classList.add('hidden');
}

export function setButtonBusy(buttonEl, isBusy, label) {
  buttonEl.disabled = Boolean(isBusy);
  if (typeof label === 'string') buttonEl.textContent = label;
}
