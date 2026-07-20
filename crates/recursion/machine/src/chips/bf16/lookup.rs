use core::borrow::{Borrow, BorrowMut};
use std::mem::MaybeUninit;

use slop_air::{Air, BaseAir, PairBuilder};
use slop_algebra::{AbstractField, Field, PrimeField32};
use slop_matrix::Matrix;
use sp1_derive::AlignedBorrow;
use sp1_hypercube::air::MachineAir;
use sp1_recursion_executor::{
    bf16_i32_to_field, bf16_lookup_row, ExecutionRecord, RecursionProgram, BF16_LOOKUP_ADD,
    BF16_LOOKUP_DIV, BF16_LOOKUP_INIT, BF16_LOOKUP_MUL, BF16_LOOKUP_RSQRT, BF16_LOOKUP_SHARED,
    BF16_LOOKUP_SQUARE, BF16_LOOKUP_TABLE_ROWS, NUM_BF16_LOOKUP_OPS,
};

use crate::builder::SP1RecursionAirBuilder;

/// The lookup contains one row for every possible 16-bit input.
pub const BF16_LOOKUP_ROWS: usize = BF16_LOOKUP_TABLE_ROWS;

pub const BF16_LOOKUP_PREPROCESSED_COLS: usize =
    core::mem::size_of::<Bf16LookupPreprocessedCols<u8>>();
pub const BF16_LOOKUP_COLS: usize = core::mem::size_of::<Bf16LookupCols<u8>>();

/// Preprocessed columns shared by BF16 initialization and arithmetic operations.
#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Bf16LookupPreprocessedCols<T: Copy> {
    pub input: T,
    pub init_sign: T,
    pub init_exponent: T,
    pub init_mantissa: T,
    pub shared: T,
    pub mul: T,
    pub div: T,
    pub add: T,
    pub square: T,
    pub rsqrt: T,
}

/// Multiplicities for every operation-specific BF16 lookup column.
#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Bf16LookupCols<T: Copy> {
    pub multiplicities: [T; NUM_BF16_LOOKUP_OPS],
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Bf16LookupChip;

impl<F: Field> BaseAir<F> for Bf16LookupChip {
    fn width(&self) -> usize {
        BF16_LOOKUP_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for Bf16LookupChip {
    type Record = ExecutionRecord<F>;
    type Program = RecursionProgram<F>;

    fn name(&self) -> &'static str {
        "Bf16Lookup"
    }

    fn num_rows(&self, _input: &Self::Record) -> Option<usize> {
        Some(BF16_LOOKUP_ROWS)
    }

    fn preprocessed_width(&self) -> usize {
        BF16_LOOKUP_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, _program: &Self::Program) -> Option<usize> {
        Some(BF16_LOOKUP_ROWS)
    }

    fn preprocessed_num_rows_with_instrs_len(
        &self,
        _program: &Self::Program,
        _instrs_len: usize,
    ) -> Option<usize> {
        Some(BF16_LOOKUP_ROWS)
    }

    fn generate_preprocessed_trace_into(
        &self,
        _program: &Self::Program,
        buffer: &mut [MaybeUninit<F>],
    ) {
        let values = unsafe {
            core::slice::from_raw_parts_mut(
                buffer.as_mut_ptr() as *mut F,
                BF16_LOOKUP_PREPROCESSED_COLS * BF16_LOOKUP_ROWS,
            )
        };

        for input in 0..=u16::MAX {
            let start = input as usize * BF16_LOOKUP_PREPROCESSED_COLS;
            let row: &mut Bf16LookupPreprocessedCols<F> =
                values[start..start + BF16_LOOKUP_PREPROCESSED_COLS].borrow_mut();
            let lookup = bf16_lookup_row(input);
            let initialized = lookup.init;
            *row = Bf16LookupPreprocessedCols {
                input: F::from_canonical_u16(input),
                init_sign: F::from_canonical_u8(initialized.sign),
                init_exponent: bf16_i32_to_field(initialized.exponent),
                init_mantissa: F::from_canonical_u16(initialized.mantissa),
                shared: bf16_i32_to_field(lookup.shared),
                mul: F::from_canonical_u16(lookup.mul),
                div: F::from_canonical_u16(lookup.div),
                add: F::from_canonical_u16(lookup.add),
                square: F::from_canonical_u16(lookup.square),
                rsqrt: F::from_canonical_u16(lookup.rsqrt),
            };
        }
    }

    fn generate_dependencies(&self, _input: &Self::Record, _output: &mut Self::Record) {}

    fn generate_trace_into(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
        buffer: &mut [MaybeUninit<F>],
    ) {
        let values = unsafe {
            core::slice::from_raw_parts_mut(
                buffer.as_mut_ptr() as *mut F,
                BF16_LOOKUP_COLS * BF16_LOOKUP_ROWS,
            )
        };
        unsafe {
            core::ptr::write_bytes(values.as_mut_ptr(), 0, values.len());
        }

        let mut increment = |opcode: u8, row: u16| {
            let index = row as usize * BF16_LOOKUP_COLS + opcode as usize;
            values[index] += F::one();
        };

        for event in &input.bf16_mul_events {
            for row in event.lookup_rows.init {
                increment(BF16_LOOKUP_INIT, row);
            }
            for row in event.lookup_rows.shared {
                increment(BF16_LOOKUP_SHARED, row);
            }
            increment(BF16_LOOKUP_MUL, event.lookup_rows.mul);
        }
        for event in &input.bf16_unary_events {
            increment(event.opcode.lookup_opcode(), event.lookup_row);
        }
        for event in &input.bf16_div_events {
            for row in event.lookup_rows.init {
                increment(BF16_LOOKUP_INIT, row);
            }
            for row in event.lookup_rows.shared {
                increment(BF16_LOOKUP_SHARED, row);
            }
            increment(BF16_LOOKUP_DIV, event.lookup_rows.div);
        }
        for event in &input.bf16_add_sub_events {
            for row in event.lookup_rows.init {
                increment(BF16_LOOKUP_INIT, row);
            }
            for row in event.lookup_rows.shared {
                increment(BF16_LOOKUP_SHARED, row);
            }
            increment(BF16_LOOKUP_ADD, event.lookup_rows.add);
        }
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }
}

impl<AB> Air<AB> for Bf16LookupChip
where
    AB: SP1RecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Bf16LookupCols<AB::Var> = (*local).borrow();

        let preprocessed = builder.preprocessed();
        let preprocessed = preprocessed.row_slice(0);
        let table: &Bf16LookupPreprocessedCols<AB::Var> = (*preprocessed).borrow();

        builder.receive_bf16_lookup(
            AB::F::from_canonical_u8(BF16_LOOKUP_INIT),
            table.input,
            table.init_sign,
            table.init_exponent,
            table.init_mantissa,
            local.multiplicities[BF16_LOOKUP_INIT as usize],
        );
        builder.receive_bf16_lookup(
            AB::F::from_canonical_u8(BF16_LOOKUP_SHARED),
            table.input,
            table.shared,
            AB::F::zero(),
            AB::F::zero(),
            local.multiplicities[BF16_LOOKUP_SHARED as usize],
        );
        builder.receive_bf16_lookup(
            AB::F::from_canonical_u8(BF16_LOOKUP_MUL),
            table.input,
            table.mul,
            AB::F::zero(),
            AB::F::zero(),
            local.multiplicities[BF16_LOOKUP_MUL as usize],
        );
        builder.receive_bf16_lookup(
            AB::F::from_canonical_u8(BF16_LOOKUP_SQUARE),
            table.input,
            table.square,
            AB::F::zero(),
            AB::F::zero(),
            local.multiplicities[BF16_LOOKUP_SQUARE as usize],
        );
        builder.receive_bf16_lookup(
            AB::F::from_canonical_u8(BF16_LOOKUP_RSQRT),
            table.input,
            table.rsqrt,
            AB::F::zero(),
            AB::F::zero(),
            local.multiplicities[BF16_LOOKUP_RSQRT as usize],
        );
        builder.receive_bf16_lookup(
            AB::F::from_canonical_u8(BF16_LOOKUP_DIV),
            table.input,
            table.div,
            AB::F::zero(),
            AB::F::zero(),
            local.multiplicities[BF16_LOOKUP_DIV as usize],
        );
        builder.receive_bf16_lookup(
            AB::F::from_canonical_u8(BF16_LOOKUP_ADD),
            table.input,
            table.add,
            AB::F::zero(),
            AB::F::zero(),
            local.multiplicities[BF16_LOOKUP_ADD as usize],
        );
    }
}
