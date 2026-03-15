/**
 * audio-processor.worklet.js
 * Place this file at: frontend/public/audio-processor.worklet.js
 *
 * Runs on the browser's dedicated audio rendering thread.
 * Converts float32 mic samples to int16 and posts each 128-sample
 * chunk immediately to useLiveAgent — no buffering, no gaps.
 */

class MicProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input || input.length === 0) return true;

    // Convert float32 [-1, 1] → int16 [-32768, 32767]
    const int16 = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      const clamped = Math.max(-1, Math.min(1, input[i]));
      int16[i] = clamped < 0 ? clamped * 32768 : clamped * 32767;
    }

    // Transfer the buffer (zero-copy) to the main thread
    this.port.postMessage(int16, [int16.buffer]);
    return true; // Keep processor alive
  }
}

registerProcessor("mic-processor", MicProcessor);