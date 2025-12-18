import { setStatus, clearStatus, setButtonBusy } from './ui.js';
import { createRenderer } from './renderer.js';
import { ApiClient } from './api.js';
import { AnimationPlayer } from './player.js';
import { buildMorphTargetIndex } from './morphTargets.js';

const MODEL_URL = '../models/facecap.glb';

function getEl(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el;
}

async function boot() {
  const container = getEl('canvas-container');
  const animateBtn = getEl('animateBtn');
  const textInput = getEl('textInput');

  setStatus('Initializing renderer…');

  const renderer = await createRenderer(container);

  setStatus('Loading 3D model…');
  const headMesh = await renderer.loadModel(MODEL_URL);

  const api = new ApiClient({ baseUrl: '' });
  const player = new AnimationPlayer({
    onStateChange: (state) => {
      if (state === 'idle') {
        setButtonBusy(animateBtn, false, 'Animate');
        clearStatus();
      }
      if (state === 'generating') {
        setButtonBusy(animateBtn, true, 'Generating…');
        setStatus('Generating animation…');
      }
      if (state === 'playing') {
        setButtonBusy(animateBtn, true, 'Playing…');
        clearStatus();
      }
      if (state === 'error') {
        setButtonBusy(animateBtn, false, 'Animate');
      }
    }
  });

  // Main render loop: keep it tiny; let player decide what to do.
  renderer.startLoop((timeMs) => {
    player.update(timeMs);
  });

  animateBtn.addEventListener('click', async () => {
    const text = String(textInput.value || '').trim();
    if (!text) return;

    try {
      player.setState('generating');

      const result = await api.animate(text);
      if (!result?.frames?.length) {
        setStatus('No frames returned from server.', { isError: true });
        player.setState('error');
        return;
      }

      // Build morph index from the *first* frame keys.
      const { indicesByName, missingNames, totalTargets } = buildMorphTargetIndex(
        headMesh,
        result.frames[0]
      );

      if (missingNames.length > 0) {
        // This is your mapping-mismatch check. If animations "work" you likely only miss a few,
        // but if you ever see a ton missing, the rig naming doesn't match the server output.
        console.warn(
          `[morph] Missing ${missingNames.length} / ${Object.keys(result.frames[0]).length} frame keys. ` +
            `Mesh targets=${totalTargets}. Example missing:`,
          missingNames.slice(0, 10)
        );
      }

      player.attach({
        headMesh,
        morphIndex: indicesByName
      });

      player.setFrames({
        frames: result.frames,
        fps: result.fps || 30
      });

      await player.playWithAudioUrl(result.audio_url);
    } catch (err) {
      console.error(err);
      const message = err?.message ? String(err.message) : 'Request failed.';
      setStatus(message, { isError: true });
      player.setState('error');
    }
  });

  clearStatus();
}

boot().catch((err) => {
  console.error(err);
  setStatus(`Fatal error: ${err?.message || err}`, { isError: true });
});
