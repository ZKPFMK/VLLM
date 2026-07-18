use std::sync::LazyLock;

use serde::{Deserialize, Serialize};
use slop_algebra::AbstractField;

/// Number of exponent bits in BF16.
pub const BF16_EXPONENT_BITS: u32 = 8;
/// Number of explicit mantissa bits in BF16.
pub const BF16_MANTISSA_BITS: u32 = 7;
/// Number of bits in a raw BF16 encoding.
pub const BF16_BITS: u32 = 16;
/// Number of rows in each 16-bit BF16 lookup column.
pub const BF16_LOOKUP_TABLE_ROWS: usize = 1 << BF16_BITS;

/// Smallest exponent of a normal BF16 value in the circuit representation.
pub const BF16_MIN_NORMAL_EXPONENT: i32 = -126;
/// Largest exponent of a normal BF16 value in the circuit representation.
pub const BF16_MAX_NORMAL_EXPONENT: i32 = 127;
/// Smallest exponent of a subnormal BF16 value in the circuit representation.
pub const BF16_MIN_SUBNORMAL_EXPONENT: i32 = -133;
/// Sentinel used for zero values in the circuit representation.
pub const BF16_ZERO_EXPONENT: i32 = -320;
/// Sentinel shared by infinities and NaNs in the circuit representation.
pub const BF16_ABNORMAL_EXPONENT: i32 = 512;

/// Lookup opcode for converting a raw BF16 value to its circuit representation.
pub const BF16_LOOKUP_INIT: u8 = 0;
/// Lookup opcode for the shared BF16 helper column.
pub const BF16_LOOKUP_SHARED: u8 = 1;
/// Lookup opcode for BF16 mantissa multiplication.
pub const BF16_LOOKUP_MUL: u8 = 2;
/// Number of lookup operations currently supplied by the BF16 table.
pub const NUM_BF16_LOOKUP_OPS: usize = 3;

/// Prefix used by the shared lookup for `Round`.
pub const BF16_ROUND_PREFIX: u16 = 0x4000;
/// Prefix used by the shared lookup for `LShift`.
pub const BF16_LSHIFT_PREFIX: u16 = 0x8000;
/// Prefix used by the shared lookup for `RShift`.
pub const BF16_RSHIFT_PREFIX: u16 = 0xc000;

/// Convert a small signed integer to a field element.
#[inline]
pub fn bf16_i32_to_field<F: AbstractField>(value: i32) -> F {
    if value >= 0 {
        F::from_canonical_u32(value as u32)
    } else {
        -F::from_canonical_u32(value.unsigned_abs())
    }
}

/// A BF16 value after the one-lookup initialization used by `VeriLLM`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct Bf16CircuitValue<T> {
    /// Raw 16-bit BF16 encoding.
    pub raw: T,
    /// Sign bit.
    pub sign: T,
    /// Unbiased exponent, or a zero/abnormal sentinel.
    pub exponent: T,
    /// Eight-bit normalized mantissa, with the implicit leading bit made explicit.
    pub mantissa: T,
}

/// Integer form of [`Bf16CircuitValue`] used by the executor and table generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bf16CircuitValueInt {
    pub raw: u16,
    pub sign: u8,
    pub exponent: i32,
    pub mantissa: u16,
}

impl Bf16CircuitValueInt {
    /// Decode a raw BF16 encoding into the representation from the `VeriLLM` paper.
    #[must_use]
    pub fn decode(raw: u16) -> Self {
        bf16_lookup_row(raw).init
    }

    /// Encode a circuit representation as a canonical raw BF16 value.
    ///
    /// `VeriLLM` intentionally merges infinity and NaN into one abnormal category. We encode that
    /// category as a signed infinity so results remain valid 16-bit BF16 values.
    #[must_use]
    pub fn encode(sign: u8, exponent: i32, mantissa: u16) -> u16 {
        debug_assert!(sign <= 1);
        let sign = (sign as u16) << 15;

        if exponent == BF16_ABNORMAL_EXPONENT {
            return sign | 0x7f80;
        }
        if exponent == BF16_ZERO_EXPONENT || mantissa == 0 {
            return sign;
        }

        if exponent >= BF16_MIN_NORMAL_EXPONENT {
            debug_assert!(exponent <= BF16_MAX_NORMAL_EXPONENT);
            debug_assert!((128..256).contains(&mantissa));
            let biased_exponent = (exponent + 127) as u16;
            sign | (biased_exponent << BF16_MANTISSA_BITS) | (mantissa - 128)
        } else {
            debug_assert!(exponent >= BF16_MIN_SUBNORMAL_EXPONENT);
            let shift = (BF16_MIN_NORMAL_EXPONENT - exponent) as u32;
            sign | (mantissa >> shift)
        }
    }

    #[must_use]
    pub fn as_field<F: AbstractField>(self) -> Bf16CircuitValue<F> {
        Bf16CircuitValue {
            raw: F::from_canonical_u16(self.raw),
            sign: F::from_canonical_u8(self.sign),
            exponent: bf16_i32_to_field(self.exponent),
            mantissa: F::from_canonical_u16(self.mantissa),
        }
    }
}

/// One row of the executor-side copy of the three BF16 lookup columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bf16LookupTableRow {
    pub init: Bf16CircuitValueInt,
    pub shared: i32,
    pub mul: u16,
}

/// The BF16 lookup table is generated once and shared by witness and trace generation.
static BF16_LOOKUP_TABLE: LazyLock<Box<[Bf16LookupTableRow]>> = LazyLock::new(|| {
    (0..=u16::MAX)
        .map(|input| Bf16LookupTableRow {
            init: evaluate_bf16_init(input),
            shared: evaluate_bf16_shared(input),
            mul: evaluate_bf16_mul(input),
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
});

/// Read all BF16 lookup columns for one 16-bit input.
#[must_use]
#[inline]
pub fn bf16_lookup_row(input: u16) -> &'static Bf16LookupTableRow {
    &BF16_LOOKUP_TABLE[input as usize]
}

/// Rows queried by one BF16 multiplication event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bf16MulLookupRows {
    /// Initialization rows for the left input, right input, and output.
    pub init: [u16; 3],
    /// Shared rows for `RShift`, `Exp`, `Clamp`, and `Round`.
    pub shared: [u16; 4],
    /// Mantissa multiplication row.
    pub mul: u16,
}

/// Complete witness for one BF16 multiplication event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct Bf16MulEvent<F> {
    pub lhs: Bf16CircuitValue<F>,
    pub rhs: Bf16CircuitValue<F>,
    pub output: Bf16CircuitValue<F>,
    /// Packed `carry || normalized_mantissa` returned by the `Mul` lookup.
    pub product: F,
    pub carry: F,
    pub normalized_mantissa: F,
    pub intermediate_exponent: F,
    /// Packed `abnormal || precision_shift` returned by `Clamp`.
    pub clamp: F,
    pub lookup_rows: Bf16MulLookupRows,
}

/// Integer witness produced by the reference implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bf16MulWitness {
    pub lhs: Bf16CircuitValueInt,
    pub rhs: Bf16CircuitValueInt,
    pub output: Bf16CircuitValueInt,
    pub product: u16,
    pub carry: u8,
    pub normalized_mantissa: u16,
    pub intermediate_exponent: i32,
    pub clamp: u16,
    pub lookup_rows: Bf16MulLookupRows,
}

impl Bf16MulWitness {
    /// Execute the lookup-based BF16 multiplication algorithm from `VeriLLM`.
    #[must_use]
    pub fn new(lhs_raw: u16, rhs_raw: u16) -> Self {
        let lhs = Bf16CircuitValueInt::decode(lhs_raw);
        let rhs = Bf16CircuitValueInt::decode(rhs_raw);

        let mul_row = (lhs.mantissa << 8) | rhs.mantissa;
        let product = bf16_mul_lookup(mul_row);

        let rshift_row = bf16_encode_rshift(1, product);
        let carry = bf16_shared_lookup(rshift_row) as u8;
        let normalized_mantissa = product - (1 << (BF16_MANTISSA_BITS + 1)) * carry as u16;

        let intermediate_exponent = lhs.exponent + rhs.exponent + carry as i32;
        let exp_row = bf16_encode_exp(intermediate_exponent);
        let output_exponent = bf16_shared_lookup(exp_row);
        let clamp_row = bf16_encode_clamp(intermediate_exponent);
        let clamp = bf16_shared_lookup(clamp_row) as u16;

        let sign = lhs.sign ^ rhs.sign;
        let round_row = bf16_encode_round(clamp, normalized_mantissa);
        let output_mantissa = bf16_shared_lookup(round_row) as u16;
        let output_raw = Bf16CircuitValueInt::encode(sign, output_exponent, output_mantissa);
        let output = Bf16CircuitValueInt::decode(output_raw);

        debug_assert_eq!(output.sign, sign);
        debug_assert_eq!(output.exponent, output_exponent);
        debug_assert_eq!(output.mantissa, output_mantissa);

        Self {
            lhs,
            rhs,
            output,
            product,
            carry,
            normalized_mantissa,
            intermediate_exponent,
            clamp,
            lookup_rows: Bf16MulLookupRows {
                init: [lhs_raw, rhs_raw, output_raw],
                shared: [rshift_row, exp_row, clamp_row, round_row],
                mul: mul_row,
            },
        }
    }

    #[must_use]
    pub fn as_event<F: AbstractField>(self) -> Bf16MulEvent<F> {
        Bf16MulEvent {
            lhs: self.lhs.as_field(),
            rhs: self.rhs.as_field(),
            output: self.output.as_field(),
            product: F::from_canonical_u16(self.product),
            carry: F::from_canonical_u8(self.carry),
            normalized_mantissa: F::from_canonical_u16(self.normalized_mantissa),
            intermediate_exponent: bf16_i32_to_field(self.intermediate_exponent),
            clamp: F::from_canonical_u16(self.clamp),
            lookup_rows: self.lookup_rows,
        }
    }
}

/// Look up the `VeriLLM` mantissa-multiplication column.
#[must_use]
#[inline]
pub fn bf16_mul_lookup(input: u16) -> u16 {
    bf16_lookup_row(input).mul
}

fn evaluate_bf16_init(raw: u16) -> Bf16CircuitValueInt {
    let sign = (raw >> 15) as u8;
    let biased_exponent = ((raw >> BF16_MANTISSA_BITS) & 0xff) as u8;
    let fraction = raw & 0x7f;

    let (exponent, mantissa) = match (biased_exponent, fraction) {
        (0, 0) => (BF16_ZERO_EXPONENT, 0),
        (0, fraction) => {
            let bit_length = u16::BITS - fraction.leading_zeros();
            let shift = BF16_MANTISSA_BITS + 1 - bit_length;
            (BF16_MIN_NORMAL_EXPONENT - shift as i32, fraction << shift)
        }
        (0xff, _) => (BF16_ABNORMAL_EXPONENT, 0),
        (biased_exponent, fraction) => {
            (biased_exponent as i32 - 127, (1 << BF16_MANTISSA_BITS) + fraction)
        }
    };

    Bf16CircuitValueInt { raw, sign, exponent, mantissa }
}

fn evaluate_bf16_mul(input: u16) -> u16 {
    let lhs = (input >> 8) as u32;
    let rhs = (input & 0xff) as u32;
    let product = lhs * rhs;
    let (carry, normalized) = if product >= (1 << (2 * BF16_MANTISSA_BITS + 1)) {
        (1, product >> (BF16_MANTISSA_BITS + 1))
    } else {
        (0, product >> BF16_MANTISSA_BITS)
    };
    ((carry << (BF16_MANTISSA_BITS + 1)) | normalized) as u16
}

/// Look up the shared 16-bit column from the `VeriLLM` paper.
#[must_use]
#[inline]
pub fn bf16_shared_lookup(input: u16) -> i32 {
    bf16_lookup_row(input).shared
}

fn evaluate_bf16_shared(input: u16) -> i32 {
    match input >> 14 {
        // Exp or Clamp: 00 || b || (v + 2^12).
        0b00 => {
            let clamp = ((input >> 13) & 1) != 0;
            let value = (input & 0x1fff) as i32 - (1 << 12);
            if clamp {
                let abnormal = i32::from(value > BF16_MAX_NORMAL_EXPONENT);
                let shift =
                    (BF16_MIN_NORMAL_EXPONENT - value).clamp(0, BF16_MANTISSA_BITS as i32 + 2);
                abnormal * 16 + shift
            } else if value < BF16_MIN_SUBNORMAL_EXPONENT {
                BF16_ZERO_EXPONENT
            } else if value > BF16_MAX_NORMAL_EXPONENT {
                BF16_ABNORMAL_EXPONENT
            } else {
                value
            }
        }
        // Round: 01 || 0 || a || l || u.
        0b01 => {
            let abnormal = ((input >> 12) & 1) != 0;
            let shift = ((input >> 8) & 0xf) as u32;
            let mantissa = input & 0xff;
            if abnormal || shift > BF16_MANTISSA_BITS {
                0
            } else {
                ((mantissa >> shift) << shift) as i32
            }
        }
        // LShift: 10 || l || s || m.
        0b10 => {
            let shift = ((input >> 10) & 0xf) as u32;
            let sign = ((input >> 9) & 1) as i32;
            let mantissa = (input & 0x1ff) as i32;
            (1 - 2 * sign) * (mantissa << shift)
        }
        // RShift: 11 || l || x.
        0b11 => {
            let offset = ((input >> 12) & 0x3) as u32;
            let value = input & 0x0fff;
            (value >> (BF16_MANTISSA_BITS + offset)) as i32
        }
        _ => unreachable!(),
    }
}

#[must_use]
pub fn bf16_encode_exp(value: i32) -> u16 {
    debug_assert!((-4096..4096).contains(&value));
    (value + (1 << 12)) as u16
}

#[must_use]
pub fn bf16_encode_clamp(value: i32) -> u16 {
    (1 << 13) | bf16_encode_exp(value)
}

#[must_use]
pub fn bf16_encode_round(clamp: u16, mantissa: u16) -> u16 {
    debug_assert!(clamp < 32);
    debug_assert!(mantissa < 256);
    BF16_ROUND_PREFIX | (clamp << 8) | mantissa
}

#[must_use]
pub fn bf16_encode_rshift(offset: u16, value: u16) -> u16 {
    debug_assert!(offset < 4);
    debug_assert!(value < 4096);
    BF16_RSHIFT_PREFIX | (offset << 12) | value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_table_contains_all_generated_rows() {
        assert_eq!(BF16_LOOKUP_TABLE.len(), BF16_LOOKUP_TABLE_ROWS);
        for input in 0..=u16::MAX {
            let row = bf16_lookup_row(input);
            assert_eq!(row.init, evaluate_bf16_init(input));
            assert_eq!(row.shared, evaluate_bf16_shared(input));
            assert_eq!(row.mul, evaluate_bf16_mul(input));
        }
    }

    #[test]
    fn decode_encode_all_finite_values() {
        for raw in 0..=u16::MAX {
            let value = Bf16CircuitValueInt::decode(raw);
            if value.exponent != BF16_ABNORMAL_EXPONENT {
                assert_eq!(
                    Bf16CircuitValueInt::encode(value.sign, value.exponent, value.mantissa),
                    raw
                );
            }
        }
    }

    #[test]
    fn multiplication_examples() {
        // 1.5 * -2.0 = -3.0.
        assert_eq!(Bf16MulWitness::new(0x3fc0, 0xc000).output.raw, 0xc040);
        // Smallest subnormal times one remains unchanged.
        assert_eq!(Bf16MulWitness::new(0x0001, 0x3f80).output.raw, 0x0001);
        // Overflow is represented canonically as infinity.
        assert_eq!(Bf16MulWitness::new(0x7f7f, 0x4000).output.raw, 0x7f80);
    }
}
