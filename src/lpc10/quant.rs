// Copyright 2026 Mark AB (markik)
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Quantization and bit packing: parameters to seven bytes and back.
//!
//! Bit budget per 22.5 ms frame, following LPC-10e's published allocation:
//!
//! | Field | Bits |
//! |---|---|
//! | voicing | 1 |
//! | pitch period | 6 |
//! | residual gain | 5 |
//! | reflection coefficients 1-4 | 5 each |
//! | reflection coefficients 5-8 | 4 each |
//! | reflection coefficient 9 | 3 |
//! | reflection coefficient 10 | 2 |
//!
//! That is 53 bits, carried in 7 bytes. The spare 3 bits are written as
//! zero and ignored on read, which leaves room to say something later
//! without moving any existing field.
//!
//! The quantizer *tables* are this crate's own. The structure above is the
//! standard's; the levels are not, so a Pipit frame is not interchangeable
//! with a FIPS-137 frame.

use super::{FRAME_BYTES, ORDER, PITCH_MAX, PITCH_MIN, Params};
use crate::math;

/// Bits given to each reflection coefficient, lowest order first. Early
/// coefficients carry the formants that decide intelligibility, so they get
/// the resolution.
const RC_BITS: [u32; ORDER] = [5, 5, 5, 5, 4, 4, 4, 4, 3, 2];

/// The first two coefficients are quantized as log area ratios, where equal
/// steps matter equally to the ear. This bounds that domain.
const LAR_LIMIT: f32 = 8.0;

/// Direct-quantized coefficients stay inside this, which keeps every
/// reconstructed filter stable.
const RC_LIMIT: f32 = 0.99;

/// Gain is logarithmic across this many powers of two, starting here. Index
/// zero is reserved for silence.
const GAIN_MIN_LOG2: f32 = -1.0;
const GAIN_MAX_LOG2: f32 = 14.0;

/// Log area ratio of a reflection coefficient.
fn to_lar(k: f32) -> f32 {
    let k = k.clamp(-0.999, 0.999);
    math::log2((1.0 + k) / (1.0 - k))
}

/// Back from a log area ratio. The result is always inside (-1, 1).
fn from_lar(lar: f32) -> f32 {
    let p = math::exp2(lar);
    (p - 1.0) / (p + 1.0)
}

/// Quantize `value` in `[lo, hi]` to `bits`, returning the code.
fn quantize(value: f32, lo: f32, hi: f32, bits: u32) -> u32 {
    let levels = (1u32 << bits) - 1;
    let normalized = (value - lo) / (hi - lo);
    let scaled = normalized * levels as f32 + 0.5;
    (scaled.clamp(0.0, levels as f32)) as u32
}

/// The inverse of [`quantize`].
fn dequantize(code: u32, lo: f32, hi: f32, bits: u32) -> f32 {
    let levels = (1u32 << bits) - 1;
    lo + (code.min(levels) as f32 / levels as f32) * (hi - lo)
}

/// Writes fields most significant bit first.
struct BitWriter {
    bytes: [u8; FRAME_BYTES],
    position: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: [0; FRAME_BYTES],
            position: 0,
        }
    }

    fn write(&mut self, value: u32, bits: u32) {
        for i in (0..bits).rev() {
            if (value >> i) & 1 == 1 {
                self.bytes[self.position / 8] |= 0x80 >> (self.position % 8);
            }
            self.position += 1;
        }
    }
}

/// Reads fields back in the same order.
struct BitReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn read(&mut self, bits: u32) -> u32 {
        let mut value = 0;
        for _ in 0..bits {
            let byte = self.bytes[self.position / 8];
            let bit = (byte >> (7 - self.position % 8)) & 1;
            value = (value << 1) | bit as u32;
            self.position += 1;
        }
        value
    }
}

/// Pack analysed parameters into one frame.
pub fn pack(params: &Params) -> [u8; FRAME_BYTES] {
    let mut writer = BitWriter::new();
    writer.write(u32::from(params.voiced), 1);

    // Pitch is logarithmic: a semitone matters the same at the top of the
    // range as at the bottom.
    let period = (params.pitch as f32).clamp(PITCH_MIN as f32, PITCH_MAX as f32);
    let pitch_code = quantize(
        math::log2(period),
        math::log2(PITCH_MIN as f32),
        math::log2(PITCH_MAX as f32),
        6,
    );
    writer.write(pitch_code, 6);

    let gain_code = if params.gain <= 0.0 {
        0
    } else {
        // Level 0 means silence, so real gains start at 1.
        let code = quantize(math::log2(params.gain), GAIN_MIN_LOG2, GAIN_MAX_LOG2, 5);
        code.max(1)
    };
    writer.write(gain_code, 5);

    for (i, &k) in params.rc.iter().enumerate() {
        let bits = RC_BITS[i];
        let code = if i < 2 {
            quantize(to_lar(k), -LAR_LIMIT, LAR_LIMIT, bits)
        } else {
            quantize(k, -RC_LIMIT, RC_LIMIT, bits)
        };
        writer.write(code, bits);
    }
    writer.bytes
}

/// Unpack one frame.
///
/// Every bit pattern is a valid frame: fields are clamped to their declared
/// ranges and reflection coefficients land strictly inside (-1, 1), so a
/// corrupted or hostile frame produces a stable filter rather than an
/// exploding one. `bytes` must be at least [`FRAME_BYTES`] long.
pub fn unpack(bytes: &[u8]) -> Params {
    debug_assert!(bytes.len() >= FRAME_BYTES);
    let mut reader = BitReader::new(bytes);
    let voiced = reader.read(1) == 1;

    let pitch_code = reader.read(6);
    let period = math::exp2(dequantize(
        pitch_code,
        math::log2(PITCH_MIN as f32),
        math::log2(PITCH_MAX as f32),
        6,
    ));
    let pitch = (period + 0.5).clamp(PITCH_MIN as f32, PITCH_MAX as f32) as u8;

    let gain_code = reader.read(5);
    let gain = if gain_code == 0 {
        0.0
    } else {
        math::exp2(dequantize(gain_code, GAIN_MIN_LOG2, GAIN_MAX_LOG2, 5))
    };

    let mut rc = [0.0f32; ORDER];
    for (i, slot) in rc.iter_mut().enumerate() {
        let bits = RC_BITS[i];
        let code = reader.read(bits);
        *slot = if i < 2 {
            from_lar(dequantize(code, -LAR_LIMIT, LAR_LIMIT, bits))
        } else {
            dequantize(code, -RC_LIMIT, RC_LIMIT, bits)
        };
    }

    Params {
        voiced,
        pitch,
        gain,
        rc,
    }
}
