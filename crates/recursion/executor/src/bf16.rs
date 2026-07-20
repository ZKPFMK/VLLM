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
/// Lookup opcode for BF16 mantissa division.
pub const BF16_LOOKUP_DIV: u8 = 3;
/// Lookup opcode for BF16 mantissa addition and normalization.
pub const BF16_LOOKUP_ADD: u8 = 4;
/// Lookup opcode for squaring a raw BF16 value.
pub const BF16_LOOKUP_SQUARE: u8 = 5;
/// Lookup opcode for the reciprocal square root of a raw BF16 value.
pub const BF16_LOOKUP_RSQRT: u8 = 6;
/// Number of lookup operations currently supplied by the BF16 table.
pub const NUM_BF16_LOOKUP_OPS: usize = 7;

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

/// One row of the executor-side copy of the BF16 lookup columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bf16LookupTableRow {
    pub init: Bf16CircuitValueInt,
    pub shared: i32,
    pub mul: u16,
    pub div: u16,
    pub add: u16,
    pub square: u16,
    pub rsqrt: u16,
}

/// The BF16 lookup table is generated once and shared by witness and trace generation.
static BF16_LOOKUP_TABLE: LazyLock<Box<[Bf16LookupTableRow]>> = LazyLock::new(|| {
    (0..=u16::MAX)
        .map(|input| Bf16LookupTableRow {
            init: evaluate_bf16_init(input),
            shared: evaluate_bf16_shared(input),
            mul: evaluate_bf16_mul(input),
            div: evaluate_bf16_div(input),
            add: evaluate_bf16_add(input),
            square: evaluate_bf16_square(input),
            rsqrt: evaluate_bf16_rsqrt(input),
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

/// Opcode for a raw-to-raw unary BF16 lookup instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Bf16UnaryOpcode {
    Square = BF16_LOOKUP_SQUARE,
    Rsqrt = BF16_LOOKUP_RSQRT,
}

impl Bf16UnaryOpcode {
    #[must_use]
    pub const fn lookup_opcode(self) -> u8 {
        self as u8
    }
}

/// Complete event for one raw-to-raw unary BF16 lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct Bf16UnaryEvent<F> {
    pub opcode: Bf16UnaryOpcode,
    pub input: F,
    pub output: F,
    /// The lookup-table row, equal to the raw input encoding.
    pub lookup_row: u16,
}

/// Integer witness produced by a raw-to-raw unary BF16 table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bf16UnaryWitness {
    pub opcode: Bf16UnaryOpcode,
    pub input: u16,
    pub output: u16,
    pub lookup_row: u16,
}

impl Bf16UnaryWitness {
    /// Execute one raw-to-raw unary BF16 lookup.
    #[must_use]
    pub fn new(opcode: Bf16UnaryOpcode, input: u16) -> Self {
        let output = match opcode {
            Bf16UnaryOpcode::Square => bf16_square_lookup(input),
            Bf16UnaryOpcode::Rsqrt => bf16_rsqrt_lookup(input),
        };
        Self { opcode, input, output, lookup_row: input }
    }

    #[must_use]
    pub fn as_event<F: AbstractField>(self) -> Bf16UnaryEvent<F> {
        Bf16UnaryEvent {
            opcode: self.opcode,
            input: F::from_canonical_u16(self.input),
            output: F::from_canonical_u16(self.output),
            lookup_row: self.lookup_row,
        }
    }
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
            clamp: F::from_canonical_u16(self.clamp),
            lookup_rows: self.lookup_rows,
        }
    }
}

/// Rows queried by one BF16 division event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bf16DivLookupRows {
    /// Initialization rows for the numerator, denominator, and output.
    pub init: [u16; 3],
    /// Shared rows for the two `RShift` queries, `Exp`, `Clamp`, and `Round`.
    pub shared: [u16; 5],
    /// Mantissa division row.
    pub div: u16,
}

/// Complete witness for one BF16 division event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct Bf16DivEvent<F> {
    pub lhs: Bf16CircuitValue<F>,
    pub rhs: Bf16CircuitValue<F>,
    pub output: Bf16CircuitValue<F>,
    /// Packed `quotient_shift || normalized_mantissa` returned by the `Div` lookup.
    pub quotient: F,
    pub quotient_shift: F,
    pub denominator_is_zero: F,
    /// Packed `abnormal || precision_shift` returned by `Clamp`.
    pub clamp: F,
    pub lookup_rows: Bf16DivLookupRows,
}

/// Integer witness for Algorithm 2, the lookup-based `VeriLLM` BF16 division.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bf16DivWitness {
    pub lhs: Bf16CircuitValueInt,
    pub rhs: Bf16CircuitValueInt,
    pub output: Bf16CircuitValueInt,
    pub quotient: u16,
    pub quotient_shift: u8,
    pub denominator_is_zero: u8,
    pub clamp: u16,
    pub lookup_rows: Bf16DivLookupRows,
}

impl Bf16DivWitness {
    /// Execute Algorithm 2 from the `VeriLLM` paper.
    #[must_use]
    pub fn new(lhs_raw: u16, rhs_raw: u16) -> Self {
        let lhs = Bf16CircuitValueInt::decode(lhs_raw);
        let rhs = Bf16CircuitValueInt::decode(rhs_raw);

        let div_row = (lhs.mantissa << 8) | rhs.mantissa;
        let quotient = bf16_div_lookup(div_row);

        let quotient_rshift_row = bf16_encode_rshift(1, quotient);
        let quotient_shift = bf16_shared_lookup(quotient_rshift_row) as u8;
        let normalized_mantissa =
            quotient - (1 << (BF16_MANTISSA_BITS + 1)) * quotient_shift as u16;

        let denominator_rshift_row = bf16_encode_rshift(0, rhs.mantissa);
        let denominator_is_nonzero = bf16_shared_lookup(denominator_rshift_row) as u8;
        let denominator_is_zero = 1 - denominator_is_nonzero;

        let intermediate_exponent = lhs.exponent - rhs.exponent - quotient_shift as i32
            + 2 * BF16_ABNORMAL_EXPONENT * denominator_is_zero as i32;
        let exp_row = bf16_encode_exp(intermediate_exponent);
        let output_exponent = bf16_shared_lookup(exp_row);
        let clamp_row = bf16_encode_clamp(intermediate_exponent);
        let clamp = bf16_shared_lookup(clamp_row) as u16;

        let sign = lhs.sign ^ rhs.sign;
        let round_row = bf16_encode_round(clamp, normalized_mantissa);
        let output_mantissa = bf16_shared_lookup(round_row) as u16;
        let output_raw = Bf16CircuitValueInt::encode(sign, output_exponent, output_mantissa);
        let output = Bf16CircuitValueInt::decode(output_raw);

        debug_assert!(quotient_shift <= 1);
        debug_assert!(denominator_is_nonzero <= 1);
        debug_assert_eq!(output.sign, sign);
        debug_assert_eq!(output.exponent, output_exponent);
        debug_assert_eq!(output.mantissa, output_mantissa);

        Self {
            lhs,
            rhs,
            output,
            quotient,
            quotient_shift,
            denominator_is_zero,
            clamp,
            lookup_rows: Bf16DivLookupRows {
                init: [lhs_raw, rhs_raw, output_raw],
                shared: [
                    quotient_rshift_row,
                    denominator_rshift_row,
                    exp_row,
                    clamp_row,
                    round_row,
                ],
                div: div_row,
            },
        }
    }

    #[must_use]
    pub fn as_event<F: AbstractField>(self) -> Bf16DivEvent<F> {
        Bf16DivEvent {
            lhs: self.lhs.as_field(),
            rhs: self.rhs.as_field(),
            output: self.output.as_field(),
            quotient: F::from_canonical_u16(self.quotient),
            quotient_shift: F::from_canonical_u8(self.quotient_shift),
            denominator_is_zero: F::from_canonical_u8(self.denominator_is_zero),
            clamp: F::from_canonical_u16(self.clamp),
            lookup_rows: self.lookup_rows,
        }
    }
}

/// Operation selected by the unified Algorithm 3 BF16 addition/subtraction chip.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Bf16AddSubOpcode {
    Add = 0,
    Sub = 1,
}

impl Bf16AddSubOpcode {
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    #[must_use]
    pub const fn effective_rhs_sign(self, rhs_sign: u8) -> u8 {
        rhs_sign ^ self.as_u8()
    }
}

/// Rows queried by one BF16 addition or subtraction event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bf16AddSubLookupRows {
    /// Initialization rows for the left input, right input, and output.
    pub init: [u16; 3],
    /// Shared rows for exponent ordering, alignment, shifts, exponent mapping, and abnormality.
    pub shared: [u16; 9],
    /// Mantissa addition and normalization row.
    pub add: u16,
}

/// Complete event for Algorithm 3 BF16 addition/subtraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct Bf16AddSubEvent<F> {
    pub lhs: Bf16CircuitValue<F>,
    pub rhs: Bf16CircuitValue<F>,
    pub output: Bf16CircuitValue<F>,
    pub effective_rhs_sign: F,
    pub exponent_order: F,
    pub larger_sign: F,
    pub larger_exponent: F,
    pub larger_mantissa: F,
    pub alignment: F,
    pub smaller_nonzero: F,
    pub max_alignment: F,
    pub selected_smaller: F,
    pub shifted_larger: F,
    pub signed_smaller: F,
    pub unsigned_sum: F,
    pub add_result: F,
    pub normalization_shift: F,
    pub normalized_nonzero: F,
    pub abnormal: F,
    pub lookup_rows: Bf16AddSubLookupRows,
}

/// Integer witness for Algorithm 3 BF16 addition/subtraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bf16AddSubWitness {
    pub lhs: Bf16CircuitValueInt,
    pub rhs: Bf16CircuitValueInt,
    pub output: Bf16CircuitValueInt,
    pub effective_rhs_sign: u8,
    pub exponent_order: u8,
    pub larger_sign: u8,
    pub larger_exponent: i32,
    pub larger_mantissa: u16,
    pub alignment: u16,
    pub smaller_nonzero: u8,
    pub max_alignment: u8,
    pub selected_smaller: u16,
    pub shifted_larger: i32,
    pub signed_smaller: i32,
    pub unsigned_sum: u16,
    pub add_result: u16,
    pub normalization_shift: u8,
    pub normalized_nonzero: u8,
    pub abnormal: u8,
    pub lookup_rows: Bf16AddSubLookupRows,
}

impl Bf16AddSubWitness {
    /// Execute Algorithm 3 from the `VeriLLM` paper.
    #[must_use]
    pub fn new(lhs_raw: u16, rhs_raw: u16, opcode: Bf16AddSubOpcode) -> Self {
        let lhs = Bf16CircuitValueInt::decode(lhs_raw);
        let rhs = Bf16CircuitValueInt::decode(rhs_raw);
        let effective_rhs_sign = opcode.effective_rhs_sign(rhs.sign);

        let exponent_order_input = (1 << 10) + (rhs.exponent - lhs.exponent - 1);
        let exponent_order_row = bf16_encode_rshift(3, exponent_order_input as u16);
        let exponent_order = bf16_shared_lookup(exponent_order_row) as u8;

        let (
            larger_sign,
            smaller_sign,
            larger_exponent,
            smaller_exponent,
            larger_mantissa,
            smaller_mantissa,
        ) = if exponent_order == 1 {
            (effective_rhs_sign, lhs.sign, rhs.exponent, lhs.exponent, rhs.mantissa, lhs.mantissa)
        } else {
            (lhs.sign, effective_rhs_sign, lhs.exponent, rhs.exponent, lhs.mantissa, rhs.mantissa)
        };

        let alignment_input = BF16_MIN_NORMAL_EXPONENT - larger_exponent + smaller_exponent;
        let alignment_row = bf16_encode_clamp(alignment_input);
        let alignment = bf16_shared_lookup(alignment_row) as u16;

        let smaller_nonzero_row = bf16_encode_rshift(0, smaller_mantissa);
        let smaller_nonzero = bf16_shared_lookup(smaller_nonzero_row) as u8;
        let max_alignment_input = (1 << (BF16_MANTISSA_BITS - 4)) * (alignment + 7);
        let max_alignment_row = bf16_encode_rshift(0, max_alignment_input);
        let max_alignment = bf16_shared_lookup(max_alignment_row) as u8;

        let selected_smaller =
            if max_alignment == 1 { smaller_nonzero as u16 } else { smaller_mantissa };
        let shift = alignment - max_alignment as u16;

        let shifted_larger_row = bf16_encode_lshift(shift, larger_sign, larger_mantissa);
        let shifted_larger = bf16_shared_lookup(shifted_larger_row);
        let signed_smaller =
            if smaller_sign == 1 { -(selected_smaller as i32) } else { selected_smaller as i32 };
        let signed_sum = shifted_larger + signed_smaller;

        // IEEE roundTowardZero chooses +0 for exact cancellation, except when both effective
        // operands are negative zero.
        let output_sign = u8::from(
            signed_sum < 0 || (signed_sum == 0 && lhs.sign == 1 && effective_rhs_sign == 1),
        );
        let unsigned_sum = signed_sum.unsigned_abs() as u16;

        let add_row = unsigned_sum;
        let add_result = bf16_add_lookup(add_row);
        let normalization_shift_row = bf16_encode_rshift(1, add_result);
        let normalization_shift = bf16_shared_lookup(normalization_shift_row) as u8;
        let normalized_mantissa =
            add_result - (1 << (BF16_MANTISSA_BITS + 1)) * normalization_shift as u16;

        let normalized_nonzero_row = bf16_encode_rshift(0, normalized_mantissa);
        let normalized_nonzero = bf16_shared_lookup(normalized_nonzero_row) as u8;
        let intermediate_exponent = larger_exponent + BF16_MANTISSA_BITS as i32 + 1
            - normalization_shift as i32
            - shift as i32
            + BF16_ZERO_EXPONENT * (1 - normalized_nonzero as i32);

        let exp_row = bf16_encode_exp(intermediate_exponent);
        let output_exponent = bf16_shared_lookup(exp_row);
        let abnormal_row = bf16_encode_rshift(2, (output_exponent - BF16_ZERO_EXPONENT) as u16);
        let abnormal = bf16_shared_lookup(abnormal_row) as u8;
        let output_mantissa = normalized_mantissa * (1 - abnormal as u16);
        let output_raw = Bf16CircuitValueInt::encode(output_sign, output_exponent, output_mantissa);
        let output = Bf16CircuitValueInt::decode(output_raw);

        debug_assert!(exponent_order <= 1);
        debug_assert!(alignment <= BF16_MANTISSA_BITS as u16 + 2);
        debug_assert!(smaller_nonzero <= 1);
        debug_assert!(max_alignment <= 1);
        debug_assert!(normalization_shift <= 2 * BF16_MANTISSA_BITS as u8 + 1);
        debug_assert!(normalized_nonzero <= 1);
        debug_assert!(abnormal <= 1);
        debug_assert_eq!(output.sign, output_sign);
        debug_assert_eq!(output.exponent, output_exponent);
        debug_assert_eq!(output.mantissa, output_mantissa);

        Self {
            lhs,
            rhs,
            output,
            effective_rhs_sign,
            exponent_order,
            larger_sign,
            larger_exponent,
            larger_mantissa,
            alignment,
            smaller_nonzero,
            max_alignment,
            selected_smaller,
            shifted_larger,
            signed_smaller,
            unsigned_sum,
            add_result,
            normalization_shift,
            normalized_nonzero,
            abnormal,
            lookup_rows: Bf16AddSubLookupRows {
                init: [lhs_raw, rhs_raw, output_raw],
                shared: [
                    exponent_order_row,
                    alignment_row,
                    smaller_nonzero_row,
                    max_alignment_row,
                    shifted_larger_row,
                    normalization_shift_row,
                    normalized_nonzero_row,
                    exp_row,
                    abnormal_row,
                ],
                add: add_row,
            },
        }
    }

    #[must_use]
    pub fn as_event<F: AbstractField>(self) -> Bf16AddSubEvent<F> {
        Bf16AddSubEvent {
            lhs: self.lhs.as_field(),
            rhs: self.rhs.as_field(),
            output: self.output.as_field(),
            effective_rhs_sign: F::from_canonical_u8(self.effective_rhs_sign),
            exponent_order: F::from_canonical_u8(self.exponent_order),
            larger_sign: F::from_canonical_u8(self.larger_sign),
            larger_exponent: bf16_i32_to_field(self.larger_exponent),
            larger_mantissa: F::from_canonical_u16(self.larger_mantissa),
            alignment: F::from_canonical_u16(self.alignment),
            smaller_nonzero: F::from_canonical_u8(self.smaller_nonzero),
            max_alignment: F::from_canonical_u8(self.max_alignment),
            selected_smaller: F::from_canonical_u16(self.selected_smaller),
            shifted_larger: bf16_i32_to_field(self.shifted_larger),
            signed_smaller: bf16_i32_to_field(self.signed_smaller),
            unsigned_sum: F::from_canonical_u16(self.unsigned_sum),
            add_result: F::from_canonical_u16(self.add_result),
            normalization_shift: F::from_canonical_u8(self.normalization_shift),
            normalized_nonzero: F::from_canonical_u8(self.normalized_nonzero),
            abnormal: F::from_canonical_u8(self.abnormal),
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

/// Look up the `VeriLLM` mantissa-division column.
#[must_use]
#[inline]
pub fn bf16_div_lookup(input: u16) -> u16 {
    bf16_lookup_row(input).div
}

/// Look up the Algorithm 3 mantissa-addition and normalization column.
#[must_use]
#[inline]
pub fn bf16_add_lookup(input: u16) -> u16 {
    bf16_lookup_row(input).add
}

/// Look up the raw BF16 encoding of `input * input`.
#[must_use]
#[inline]
pub fn bf16_square_lookup(input: u16) -> u16 {
    bf16_lookup_row(input).square
}

/// Look up the raw BF16 encoding of `1 / sqrt(input)`.
#[must_use]
#[inline]
pub fn bf16_rsqrt_lookup(input: u16) -> u16 {
    bf16_lookup_row(input).rsqrt
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

fn evaluate_bf16_div(input: u16) -> u16 {
    let lhs = (input >> 8) as u32;
    let rhs = (input & 0xff) as u32;
    if lhs == 0 || rhs == 0 {
        return 0;
    }

    let (quotient_shift, normalized) = if lhs >= rhs {
        (0, (lhs << BF16_MANTISSA_BITS) / rhs)
    } else {
        (1, (lhs << (BF16_MANTISSA_BITS + 1)) / rhs)
    };
    ((quotient_shift << (BF16_MANTISSA_BITS + 1)) | normalized) as u16
}

fn evaluate_bf16_add(input: u16) -> u16 {
    if input == 0 {
        return ((2 * BF16_MANTISSA_BITS + 1) << (BF16_MANTISSA_BITS + 1)) as u16;
    }

    let floor_log2 = u16::BITS - 1 - input.leading_zeros();
    let normalization_shift = 2 * BF16_MANTISSA_BITS + 1 - floor_log2;
    let normalized = ((input as u32) << normalization_shift) >> (BF16_MANTISSA_BITS + 1);
    ((normalization_shift << (BF16_MANTISSA_BITS + 1)) | normalized) as u16
}

/// Pure table-generation implementation of `input * input`.
///
/// This deliberately uses the evaluator functions directly instead of [`Bf16MulWitness`], because
/// the witness reads this same lazily initialized table.
fn evaluate_bf16_square(input: u16) -> u16 {
    let value = evaluate_bf16_init(input);

    let mul_row = (value.mantissa << 8) | value.mantissa;
    let product = evaluate_bf16_mul(mul_row);
    let carry = evaluate_bf16_shared(bf16_encode_rshift(1, product)) as u8;
    let normalized_mantissa = product - (1 << (BF16_MANTISSA_BITS + 1)) * carry as u16;

    let intermediate_exponent = value.exponent * 2 + carry as i32;
    let output_exponent = evaluate_bf16_shared(bf16_encode_exp(intermediate_exponent));
    let clamp = evaluate_bf16_shared(bf16_encode_clamp(intermediate_exponent)) as u16;
    let output_mantissa =
        evaluate_bf16_shared(bf16_encode_round(clamp, normalized_mantissa)) as u16;

    // Squaring always clears the sign because the multiplication chip computes sign XOR sign.
    Bf16CircuitValueInt::encode(0, output_exponent, output_mantissa)
}

/// Pure table-generation implementation of `1 / sqrt(input)` rounded toward zero to BF16.
///
/// Layer normalization only queries positive finite inputs. Zero, negative, and abnormal inputs
/// are mapped to the canonical abnormal value so the lookup remains a total 16-bit function.
/// For a positive finite value `x = m * 2^(e - 7)`, the returned normalized mantissa `n` is the
/// largest integer satisfying `n^2 * m <= 2^(21 - 2f - e)`, where `f` is the output exponent.
/// This computes the exact BF16 round-toward-zero result without an intermediate host float.
fn evaluate_bf16_rsqrt(input: u16) -> u16 {
    let value = evaluate_bf16_init(input);
    if value.sign != 0
        || value.exponent == BF16_ZERO_EXPONENT
        || value.exponent == BF16_ABNORMAL_EXPONENT
    {
        return Bf16CircuitValueInt::encode(0, BF16_ABNORMAL_EXPONENT, 0);
    }

    let exponent = value.exponent;
    let output_exponent = if exponent % 2 == 0 {
        -exponent / 2 - i32::from(value.mantissa != 1 << BF16_MANTISSA_BITS)
    } else {
        -(exponent + 1) / 2
    };
    let bound_exponent = 21 - 2 * output_exponent - exponent;
    debug_assert!((21..=23).contains(&bound_exponent));
    let bound = 1_u32 << bound_exponent;
    let input_mantissa = u32::from(value.mantissa);

    let mut low = 1_u16 << BF16_MANTISSA_BITS;
    let mut high = 1_u16 << (BF16_MANTISSA_BITS + 1);
    while low + 1 < high {
        let candidate = low + (high - low) / 2;
        if u32::from(candidate).pow(2) * input_mantissa <= bound {
            low = candidate;
        } else {
            high = candidate;
        }
    }

    Bf16CircuitValueInt::encode(0, output_exponent, low)
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

#[must_use]
pub fn bf16_encode_lshift(shift: u16, sign: u8, mantissa: u16) -> u16 {
    debug_assert!(shift < 16);
    debug_assert!(sign <= 1);
    debug_assert!(mantissa < 512);
    BF16_LSHIFT_PREFIX | (shift << 10) | ((sign as u16) << 9) | mantissa
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
            assert_eq!(row.div, evaluate_bf16_div(input));
            assert_eq!(row.add, evaluate_bf16_add(input));
            assert_eq!(row.square, evaluate_bf16_square(input));
            assert_eq!(row.rsqrt, evaluate_bf16_rsqrt(input));
        }
    }

    #[test]
    fn square_table_matches_multiplication_for_every_input() {
        for input in 0..=u16::MAX {
            assert_eq!(
                Bf16UnaryWitness::new(Bf16UnaryOpcode::Square, input).output,
                Bf16MulWitness::new(input, input).output.raw,
                "BF16 square mismatch for {input:04x}"
            );
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

    #[test]
    fn square_examples() {
        // 1.5^2 = 2.25.
        assert_eq!(Bf16UnaryWitness::new(Bf16UnaryOpcode::Square, 0x3fc0).output, 0x4010);
        // Squaring clears the sign.
        assert_eq!(Bf16UnaryWitness::new(Bf16UnaryOpcode::Square, 0xc000).output, 0x4080);
        assert_eq!(Bf16UnaryWitness::new(Bf16UnaryOpcode::Square, 0x8000).output, 0x0000);
    }

    #[test]
    fn rsqrt_rounds_toward_zero_for_every_positive_finite_input() {
        for input in 0..=u16::MAX {
            let value = evaluate_bf16_init(input);
            if value.sign != 0
                || value.exponent == BF16_ZERO_EXPONENT
                || value.exponent == BF16_ABNORMAL_EXPONENT
            {
                continue;
            }

            let output = evaluate_bf16_init(bf16_rsqrt_lookup(input));
            assert_eq!(output.sign, 0);
            assert_ne!(output.exponent, BF16_ZERO_EXPONENT);
            assert_ne!(output.exponent, BF16_ABNORMAL_EXPONENT);

            let bound_exponent = 21 - 2 * output.exponent - value.exponent;
            let bound = 1_u32 << bound_exponent;
            let input_mantissa = u32::from(value.mantissa);
            let output_mantissa = u32::from(output.mantissa);
            assert!(output_mantissa.pow(2) * input_mantissa <= bound);
            assert!((output_mantissa + 1).pow(2) * input_mantissa > bound);
        }
    }

    #[test]
    fn rsqrt_examples() {
        assert_eq!(Bf16UnaryWitness::new(Bf16UnaryOpcode::Rsqrt, 0x3f80).output, 0x3f80);
        assert_eq!(Bf16UnaryWitness::new(Bf16UnaryOpcode::Rsqrt, 0x4080).output, 0x3f00);
        assert_eq!(Bf16UnaryWitness::new(Bf16UnaryOpcode::Rsqrt, 0x3e80).output, 0x4000);
        assert_eq!(Bf16UnaryWitness::new(Bf16UnaryOpcode::Rsqrt, 0x4000).output, 0x3f35);
        assert_eq!(Bf16UnaryWitness::new(Bf16UnaryOpcode::Rsqrt, 0x0000).output, 0x7f80);
        assert_eq!(Bf16UnaryWitness::new(Bf16UnaryOpcode::Rsqrt, 0xbf80).output, 0x7f80);
        assert_eq!(Bf16UnaryWitness::new(Bf16UnaryOpcode::Rsqrt, 0x7f80).output, 0x7f80);
    }

    #[test]
    fn division_examples() {
        // 3.0 / -2.0 = -1.5.
        assert_eq!(Bf16DivWitness::new(0x4040, 0xc000).output.raw, 0xbfc0);
        // Smallest subnormal divided by one remains unchanged.
        assert_eq!(Bf16DivWitness::new(0x0001, 0x3f80).output.raw, 0x0001);
        // A finite nonzero value divided by zero is abnormal.
        assert_eq!(Bf16DivWitness::new(0x3f80, 0x0000).output.raw, 0x7f80);
        // Zero divided by zero is abnormal.
        assert_eq!(Bf16DivWitness::new(0x0000, 0x0000).output.raw, 0x7f80);
    }

    #[test]
    fn addition_and_subtraction_examples() {
        assert_eq!(
            Bf16AddSubWitness::new(0x3fc0, 0x4000, Bf16AddSubOpcode::Add).output.raw,
            0x4060
        );
        assert_eq!(
            Bf16AddSubWitness::new(0x4040, 0xc000, Bf16AddSubOpcode::Add).output.raw,
            0x3f80
        );
        assert_eq!(
            Bf16AddSubWitness::new(0x4040, 0x4000, Bf16AddSubOpcode::Sub).output.raw,
            0x3f80
        );
        assert_eq!(
            Bf16AddSubWitness::new(0x3f80, 0xbf80, Bf16AddSubOpcode::Add).output.raw,
            0x0000
        );
        assert_eq!(
            Bf16AddSubWitness::new(0x8000, 0x8000, Bf16AddSubOpcode::Add).output.raw,
            0x8000
        );
    }
}
