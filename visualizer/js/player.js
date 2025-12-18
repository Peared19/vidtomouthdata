import { applyFrameFast } from './morphTargets.js';

// Minimal state machine:
//   idle -> generating -> playing -> idle
//   any  -> error -> idle (next click)

export class AnimationPlayer {
  constructor({ onStateChange } = {}) {
    this.onStateChange = typeof onStateChange === 'function' ? onStateChange : () => {};

    this.state = 'idle';
    this.frames = [];
    this.fps = 30;

    this.headMesh = null;
    this.morphIndex = null;

    this.audio = null;
    this.lastAppliedFrame = -1;
  }

  setState(state) {
    this.state = state;
    this.onStateChange(state);
  }

  attach({ headMesh, morphIndex }) {
    this.headMesh = headMesh;
    this.morphIndex = morphIndex;
  }

  setFrames({ frames, fps }) {
    this.frames = Array.isArray(frames) ? frames : [];
    this.fps = Number.isFinite(fps) ? fps : 30;
    this.lastAppliedFrame = -1;
  }

  async playWithAudioUrl(audioUrl) {
    // Stop previous audio.
    if (this.audio) {
      try {
        this.audio.pause();
      } catch {}
      this.audio = null;
    }

    if (!audioUrl) {
      // Fallback to time-based playback without audio.
      this.setState('playing');
      this._startTimeMs = performance.now();
      return;
    }

    const audio = new Audio(audioUrl);
    this.audio = audio;

    // If audio ends, return to idle.
    audio.addEventListener('ended', () => {
      this.setState('idle');
    });

    this.setState('playing');

    try {
      // Attempt to play immediately (should succeed since this is called from a click handler).
      await audio.play();
    } catch (err) {
      // Autoplay policy can still block in some cases.
      // We stay in "playing" state so frames can advance with fallback timing.
      console.warn('Audio play failed; continuing without audio sync.', err);
      this.audio = null;
      this._startTimeMs = performance.now();
    }
  }

  update(timeMs) {
    if (this.state !== 'playing') return;
    if (!this.headMesh || !this.morphIndex || !this.frames.length) return;

    const maxIndex = this.frames.length - 1;

    let frameIndex;
    if (this.audio) {
      // Primary: audio-time-derived frame index.
      const t = Math.max(0, this.audio.currentTime);
      frameIndex = Math.floor(t * this.fps);
    } else {
      // Fallback: local clock.
      const elapsedSec = (timeMs - (this._startTimeMs || timeMs)) / 1000.0;
      frameIndex = Math.floor(elapsedSec * this.fps);
    }

    if (frameIndex < 0) frameIndex = 0;
    if (frameIndex > maxIndex) frameIndex = maxIndex;

    if (frameIndex !== this.lastAppliedFrame) {
      applyFrameFast(this.headMesh, this.morphIndex, this.frames[frameIndex]);
      this.lastAppliedFrame = frameIndex;
    }

    // If we've hit the end and no audio is driving us, stop.
    if (!this.audio && frameIndex >= maxIndex) {
      this.setState('idle');
    }
  }
}
