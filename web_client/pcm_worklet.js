// AudioWorklet that resamples mic audio to 16kHz and emits Int16 frames of 20ms (320 samples).
class PcmEncoderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.outRate = 16000;
    this.frameSize = 320;
    this.resampleT = 0.0;
    this.carry = [];
  }

  _floatToInt16(f) {
    const v = Math.max(-1, Math.min(1, f));
    return v < 0 ? (v * 0x8000) : (v * 0x7fff);
  }

  process(inputs, outputs, params) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch0 = input[0];

    // Linear resample from sampleRate (AudioWorklet global) to 16kHz.
    const ratio = sampleRate / this.outRate;
    const out = [];
    let t = this.resampleT;
    while (t < ch0.length - 1) {
      const i = Math.floor(t);
      const frac = t - i;
      const s = ch0[i] * (1 - frac) + ch0[i + 1] * frac;
      out.push(s);
      t += ratio;
    }
    this.resampleT = t - (ch0.length - 1);

    if (out.length) {
      this.carry = this.carry.concat(out);
    }

    // Emit 20ms frames (320 samples at 16k)
    while (this.carry.length >= this.frameSize) {
      const frame = this.carry.slice(0, this.frameSize);
      this.carry = this.carry.slice(this.frameSize);
      const pcm16 = new Int16Array(this.frameSize);
      for (let i = 0; i < this.frameSize; i++) pcm16[i] = this._floatToInt16(frame[i]);
      // Transfer the underlying buffer to main thread (zero-copy).
      this.port.postMessage(pcm16, [pcm16.buffer]);
    }

    return true;
  }
}

registerProcessor("pcm-encoder", PcmEncoderProcessor);


