//! Core-style global memory boundaries for independently provable recursion event shards.
//!
//! Each event shard normally has an unbalanced local memory multiset because the arithmetic and
//! memory event vectors are sliced independently. Before proving, [`prepare_event_shard_boundaries`]
//! computes that imbalance and inserts one compressed opposite interaction per distinct
//! `(address, value)` message. This makes the shard's ordinary LogUp close independently.
//!
//! The same boundary row is hashed with Poseidon2. Fourteen output coordinates are accumulated
//! linearly with the boundary multiplicity and direction. The accumulator is exposed through the
//! existing `global_cumulative_sum` public-value storage. A recursive join verifies every shard
//! proof independently and constrains the coordinate-wise sum of these accumulators to zero.

use core::{borrow::Borrow, iter::once, mem::MaybeUninit};
use std::borrow::BorrowMut;

use hashbrown::{hash_map::Entry, HashMap};
use slop_air::{Air, AirBuilder, BaseAir, PairBuilder};
use slop_algebra::{AbstractField, Field, PrimeField32};
use slop_matrix::Matrix;
use slop_symmetric::Permutation;
use sp1_derive::AlignedBorrow;
use sp1_hypercube::{
    air::{AirInteraction, InteractionScope, MachineAir},
    inner_perm,
    operations::poseidon2::{
        air::{eval_external_round, eval_internal_rounds},
        permutation::Poseidon2Cols,
        trace::populate_perm_deg3,
        Poseidon2Operation, NUM_EXTERNAL_ROUNDS, WIDTH,
    },
    septic_curve::SepticCurve,
    septic_digest::SepticDigest,
    septic_extension::SepticExtension,
    InteractionKind,
};
use sp1_primitives::SP1Field;
use sp1_recursion_executor::instruction::{
    HintAddCurveInstr, HintBitsInstr, HintExt2FeltsInstr, HintInstr,
};
use sp1_recursion_executor::{
    event_shard_program_digest, Address, BaseAluIo, Bf16AddSubInstr, Bf16DivInstr, Bf16MulInstr,
    Bf16UnaryInstr, Block, ExecutionRecord, GlobalMemoryBoundaryEvent, Instruction, MemAccessKind,
    RecursionProgram, D,
};

use crate::builder::SP1RecursionAirBuilder;

/// Domain separator for the vector-valued Poseidon multiset hash.
const GLOBAL_MEMORY_HASH_DOMAIN: u32 = 0x564d_454d;
const ACCUMULATOR_WIDTH: usize = 2 * 7;

pub const GLOBAL_MEMORY_BOUNDARY_COLS: usize = core::mem::size_of::<GlobalMemoryBoundaryCols<u8>>();

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct BoundaryAccumulatorCols<T: Copy> {
    pub initial: [T; ACCUMULATOR_WIDTH],
    pub cumulative: [T; ACCUMULATOR_WIDTH],
}

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct GlobalMemoryBoundaryCols<T: Copy> {
    pub addr: T,
    pub value: Block<T>,
    /// Signed memory multiplicity: sends are positive and receives are negative.
    pub signed_multiplicity: T,
    pub is_real: T,
    pub index: T,
    pub permutation: Poseidon2Operation<T>,
    pub accumulation: BoundaryAccumulatorCols<T>,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct GlobalMemoryBoundaryChip;

fn hash_input<F: AbstractField + Clone>(addr: F, value: Block<F>) -> [F; WIDTH] {
    let mut input = core::array::from_fn(|_| F::zero());
    input[0] = F::from_canonical_u32(GLOBAL_MEMORY_HASH_DOMAIN);
    input[1] = addr;
    input[2..2 + D].clone_from_slice(&value.0);
    input
}

impl<F> BaseAir<F> for GlobalMemoryBoundaryChip {
    fn width(&self) -> usize {
        GLOBAL_MEMORY_BOUNDARY_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for GlobalMemoryBoundaryChip {
    type Record = ExecutionRecord<F>;
    type Program = RecursionProgram<F>;

    fn name(&self) -> &'static str {
        "GlobalMemoryBoundary"
    }

    fn generate_dependencies(&self, _input: &Self::Record, _output: &mut Self::Record) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(input.global_memory_boundary_events.len().next_multiple_of(32).max(16))
    }

    fn generate_trace_into(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
        buffer: &mut [MaybeUninit<F>],
    ) {
        let rows = self.num_rows(input).unwrap();
        let values = unsafe {
            core::slice::from_raw_parts_mut(
                buffer.as_mut_ptr() as *mut F,
                rows * GLOBAL_MEMORY_BOUNDARY_COLS,
            )
        };
        unsafe {
            core::ptr::write_bytes(values.as_mut_ptr(), 0, values.len());
        }

        let mut accumulator = [F::zero(); ACCUMULATOR_WIDTH];
        for row_index in 0..rows {
            let start = row_index * GLOBAL_MEMORY_BOUNDARY_COLS;
            let end = start + GLOBAL_MEMORY_BOUNDARY_COLS;
            let cols: &mut GlobalMemoryBoundaryCols<F> = values[start..end].borrow_mut();
            if let Some(event) = input.global_memory_boundary_events.get(row_index) {
                cols.addr = event.addr.0;
                cols.value = event.value;
                cols.signed_multiplicity =
                    if event.is_receive { -event.multiplicity } else { event.multiplicity };
                cols.is_real = F::one();
                cols.index = F::from_canonical_usize(row_index);
                cols.permutation = populate_perm_deg3(hash_input(event.addr.0, event.value), None);
                cols.accumulation.initial = accumulator;
                let output = cols.permutation.permutation.perm_output();
                for i in 0..ACCUMULATOR_WIDTH {
                    accumulator[i] += cols.signed_multiplicity * output[i];
                }
                cols.accumulation.cumulative = accumulator;
            } else {
                cols.permutation = populate_perm_deg3([F::zero(); WIDTH], None);
            }
        }
    }

    fn included(&self, record: &Self::Record) -> bool {
        record.program.event_ranges.is_some()
    }
}

impl<AB> Air<AB> for GlobalMemoryBoundaryChip
where
    AB: SP1RecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &GlobalMemoryBoundaryCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);
        builder.when_not(local.is_real).assert_zero(local.signed_multiplicity);

        let memory_message =
            once(local.addr.into()).chain(local.value.0.map(Into::into)).collect::<Vec<_>>();
        builder.send(
            AirInteraction::new(
                memory_message,
                local.signed_multiplicity.into(),
                InteractionKind::Memory,
            ),
            InteractionScope::Local,
        );

        let expected_input = hash_input::<AB::Expr>(local.addr.into(), local.value.map(Into::into));
        for (actual, expected) in
            local.permutation.permutation.external_rounds_state()[0].iter().zip(expected_input)
        {
            builder.when(local.is_real).assert_eq((*actual).into(), expected);
        }
        for round in 0..NUM_EXTERNAL_ROUNDS {
            eval_external_round(builder, &local.permutation.permutation, round);
        }
        eval_internal_rounds(builder, &local.permutation.permutation);

        let output = local.permutation.permutation.perm_output();
        for i in 0..ACCUMULATOR_WIDTH {
            builder.assert_eq(
                local.accumulation.cumulative[i],
                local.accumulation.initial[i] + local.signed_multiplicity * output[i],
            );
        }

        builder.receive(
            AirInteraction::new(
                once(local.index.into())
                    .chain(local.accumulation.initial.map(Into::into))
                    .collect(),
                local.is_real.into(),
                InteractionKind::GlobalAccumulation,
            ),
            InteractionScope::Local,
        );
        builder.send(
            AirInteraction::new(
                once(local.index + AB::F::one())
                    .chain(local.accumulation.cumulative.map(Into::into))
                    .collect(),
                local.is_real.into(),
                InteractionKind::GlobalAccumulation,
            ),
            InteractionScope::Local,
        );
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct MemoryKey {
    addr: u32,
    value: [u32; D],
}

fn update_balance(
    balances: &mut HashMap<MemoryKey, SP1Field>,
    addr: Address<SP1Field>,
    value: Block<SP1Field>,
    delta: SP1Field,
) {
    if delta.is_zero() {
        return;
    }
    let key = MemoryKey {
        addr: addr.0.as_canonical_u32(),
        value: value.0.map(|element| element.as_canonical_u32()),
    };
    match balances.entry(key) {
        Entry::Occupied(mut occupied) => {
            let updated = *occupied.get() + delta;
            if updated.is_zero() {
                occupied.remove();
            } else {
                *occupied.get_mut() = updated;
            }
        }
        Entry::Vacant(vacant) => {
            vacant.insert(delta);
        }
    }
}

fn update_single(
    balances: &mut HashMap<MemoryKey, SP1Field>,
    addr: Address<SP1Field>,
    value: SP1Field,
    delta: SP1Field,
) {
    update_balance(balances, addr, Block::from(value), delta);
}

fn collect_hint_outputs(
    balances: &mut HashMap<MemoryKey, SP1Field>,
    record: &ExecutionRecord<SP1Field>,
    global_start: usize,
    outputs: &[(Address<SP1Field>, SP1Field)],
) {
    let active = record.program.event_ranges().mem_var;
    let local_start = active.start.max(global_start);
    let local_end = active.end.min(global_start + outputs.len());
    for global_index in local_start..local_end {
        let (addr, mult) = outputs[global_index - global_start];
        let value = record.mem_var_events[global_index - active.start].inner;
        update_balance(balances, addr, value, mult);
    }
}

fn collect_mul(
    balances: &mut HashMap<MemoryKey, SP1Field>,
    instruction: Bf16MulInstr<SP1Field>,
    event: &sp1_recursion_executor::Bf16MulEvent<SP1Field>,
) {
    update_single(balances, instruction.addrs.lhs, event.lhs.raw, -SP1Field::one());
    update_single(balances, instruction.addrs.rhs, event.rhs.raw, -SP1Field::one());
    update_single(balances, instruction.addrs.output, event.output.raw, instruction.mult);
}

fn collect_add_sub(
    balances: &mut HashMap<MemoryKey, SP1Field>,
    instruction: Bf16AddSubInstr<SP1Field>,
    event: &sp1_recursion_executor::Bf16AddSubEvent<SP1Field>,
) {
    update_single(balances, instruction.addrs.lhs, event.lhs.raw, -SP1Field::one());
    update_single(balances, instruction.addrs.rhs, event.rhs.raw, -SP1Field::one());
    update_single(balances, instruction.addrs.output, event.output.raw, instruction.mult);
}

fn collect_div(
    balances: &mut HashMap<MemoryKey, SP1Field>,
    instruction: Bf16DivInstr<SP1Field>,
    event: &sp1_recursion_executor::Bf16DivEvent<SP1Field>,
) {
    update_single(balances, instruction.addrs.lhs, event.lhs.raw, -SP1Field::one());
    update_single(balances, instruction.addrs.rhs, event.rhs.raw, -SP1Field::one());
    update_single(balances, instruction.addrs.output, event.output.raw, instruction.mult);
}

fn collect_unary(
    balances: &mut HashMap<MemoryKey, SP1Field>,
    instruction: Bf16UnaryInstr<SP1Field>,
    event: &sp1_recursion_executor::Bf16UnaryEvent<SP1Field>,
) {
    update_single(balances, instruction.addrs.input, event.input, -SP1Field::one());
    update_single(balances, instruction.addrs.output, event.output, instruction.mult);
}

fn collect_record_memory_balances(
    record: &ExecutionRecord<SP1Field>,
) -> HashMap<MemoryKey, SP1Field> {
    let mut balances = HashMap::new();
    let ranges = record.program.event_ranges();

    for analyzed in record.program.inner.iter() {
        match analyzed.inner() {
            Instruction::Mem(instruction) => {
                let global_index = analyzed.offset();
                if global_index < ranges.mem_const.start || global_index >= ranges.mem_const.end {
                    continue;
                }
                let delta = match instruction.kind {
                    MemAccessKind::Read => -instruction.mult,
                    MemAccessKind::Write => instruction.mult,
                };
                update_balance(
                    &mut balances,
                    instruction.addrs.inner,
                    instruction.vals.inner,
                    delta,
                );
            }
            Instruction::Hint(HintInstr { output_addrs_mults }) => {
                collect_hint_outputs(&mut balances, record, analyzed.offset(), output_addrs_mults);
            }
            Instruction::HintBits(HintBitsInstr { output_addrs_mults, input_addr: _ }) => {
                collect_hint_outputs(&mut balances, record, analyzed.offset(), output_addrs_mults);
            }
            Instruction::HintExt2Felts(HintExt2FeltsInstr {
                output_addrs_mults,
                input_addr: _,
            }) => {
                collect_hint_outputs(&mut balances, record, analyzed.offset(), output_addrs_mults);
            }
            Instruction::HintAddCurve(instruction) => {
                let HintAddCurveInstr { output_x_addrs_mults, output_y_addrs_mults, .. } =
                    instruction.as_ref();
                collect_hint_outputs(
                    &mut balances,
                    record,
                    analyzed.offset(),
                    output_x_addrs_mults,
                );
                collect_hint_outputs(
                    &mut balances,
                    record,
                    analyzed.offset() + output_x_addrs_mults.len(),
                    output_y_addrs_mults,
                );
            }
            Instruction::BaseAlu(instruction) => {
                let global_index = analyzed.offset();
                if global_index < ranges.base_alu.start || global_index >= ranges.base_alu.end {
                    continue;
                }
                let BaseAluIo { out, in1, in2 } =
                    record.base_alu_events[global_index - ranges.base_alu.start];
                update_single(&mut balances, instruction.addrs.in1, in1, -SP1Field::one());
                update_single(&mut balances, instruction.addrs.in2, in2, -SP1Field::one());
                update_single(&mut balances, instruction.addrs.out, out, instruction.mult);
            }
            Instruction::Poseidon2(instruction) => {
                let global_index = analyzed.offset();
                if global_index < ranges.poseidon2_wide.start
                    || global_index >= ranges.poseidon2_wide.end
                {
                    continue;
                }
                let event = &record.poseidon2_events[global_index - ranges.poseidon2_wide.start];
                for i in 0..WIDTH {
                    update_single(
                        &mut balances,
                        instruction.addrs.input[i],
                        event.input[i],
                        -SP1Field::one(),
                    );
                    update_single(
                        &mut balances,
                        instruction.addrs.output[i],
                        event.output[i],
                        instruction.mults[i],
                    );
                }
            }
            Instruction::Bf16Mul(instruction) => {
                let global_index = analyzed.offset();
                if global_index < ranges.bf16_mul.start || global_index >= ranges.bf16_mul.end {
                    continue;
                }
                collect_mul(
                    &mut balances,
                    *instruction,
                    &record.bf16_mul_events[global_index - ranges.bf16_mul.start],
                );
            }
            Instruction::Bf16Unary(instruction) => {
                let global_index = analyzed.offset();
                if global_index < ranges.bf16_unary.start || global_index >= ranges.bf16_unary.end {
                    continue;
                }
                collect_unary(
                    &mut balances,
                    *instruction,
                    &record.bf16_unary_events[global_index - ranges.bf16_unary.start],
                );
            }
            Instruction::Bf16Div(instruction) => {
                let global_index = analyzed.offset();
                if global_index < ranges.bf16_div.start || global_index >= ranges.bf16_div.end {
                    continue;
                }
                collect_div(
                    &mut balances,
                    *instruction,
                    &record.bf16_div_events[global_index - ranges.bf16_div.start],
                );
            }
            Instruction::Bf16AddSub(instruction) => {
                let global_index = analyzed.offset();
                if global_index < ranges.bf16_add_sub.start
                    || global_index >= ranges.bf16_add_sub.end
                {
                    continue;
                }
                collect_add_sub(
                    &mut balances,
                    *instruction,
                    &record.bf16_add_sub_events[global_index - ranges.bf16_add_sub.start],
                );
            }
            Instruction::Bf16LinearBatch(batch) => {
                let mul_start = analyzed.offset();
                let add_start = analyzed.secondary_offset();
                let mul_begin = ranges.bf16_mul.start.max(mul_start);
                let mul_end = ranges.bf16_mul.end.min(mul_start + batch.mul_count());
                let add_begin = ranges.bf16_add_sub.start.max(add_start);
                let add_end = ranges.bf16_add_sub.end.min(add_start + batch.add_sub_count());
                let mut dot_start = batch.dot_count();
                let mut dot_end = 0usize;
                if mul_begin < mul_end {
                    dot_start = dot_start.min((mul_begin - mul_start) / batch.input_features);
                    dot_end = dot_end.max((mul_end - mul_start).div_ceil(batch.input_features));
                }
                if add_begin < add_end {
                    let additions_per_dot = batch.input_features - 1;
                    dot_start = dot_start.min((add_begin - add_start) / additions_per_dot);
                    dot_end = dot_end.max((add_end - add_start).div_ceil(additions_per_dot));
                }
                for dot in dot_start..dot_end {
                    for input in 0..batch.input_features {
                        let mul_index = dot * batch.input_features + input;
                        let global_mul = mul_start + mul_index;
                        if ranges.bf16_mul.start <= global_mul && global_mul < ranges.bf16_mul.end {
                            collect_mul(
                                &mut balances,
                                batch.mul_instruction(mul_index),
                                &record.bf16_mul_events[global_mul - ranges.bf16_mul.start],
                            );
                        }
                        if input > 0 {
                            let add_index = dot * (batch.input_features - 1) + input - 1;
                            let global_add = add_start + add_index;
                            if ranges.bf16_add_sub.start <= global_add
                                && global_add < ranges.bf16_add_sub.end
                            {
                                collect_add_sub(
                                    &mut balances,
                                    batch.add_sub_instruction(add_index),
                                    &record.bf16_add_sub_events
                                        [global_add - ranges.bf16_add_sub.start],
                                );
                            }
                        }
                    }
                }
            }
            Instruction::Bf16MeanBatch(batch) => {
                let add_start = analyzed.offset();
                for index in 0..batch.add_sub_count() {
                    let global_index = add_start + index;
                    if ranges.bf16_add_sub.start <= global_index
                        && global_index < ranges.bf16_add_sub.end
                    {
                        collect_add_sub(
                            &mut balances,
                            batch.add_sub_instruction(index),
                            &record.bf16_add_sub_events[global_index - ranges.bf16_add_sub.start],
                        );
                    }
                }
                let global_index = analyzed.secondary_offset();
                if ranges.bf16_mul.start <= global_index && global_index < ranges.bf16_mul.end {
                    collect_mul(
                        &mut balances,
                        batch.mul_instruction(),
                        &record.bf16_mul_events[global_index - ranges.bf16_mul.start],
                    );
                }
            }
            Instruction::CommitPublicValues(_)
            | Instruction::ExtAlu(_)
            | Instruction::ExtFelt(_)
            | Instruction::Poseidon2LinearLayer(_)
            | Instruction::Poseidon2SBox(_)
            | Instruction::Select(_)
            | Instruction::PrefixSumChecks(_)
            | Instruction::Print(_)
            | Instruction::DebugBacktrace(_) => {}
        }
    }
    balances
}

fn boundary_accumulator(events: &[GlobalMemoryBoundaryEvent<SP1Field>]) -> [SP1Field; 14] {
    let mut accumulator = [SP1Field::zero(); 14];
    for event in events {
        let mut state = hash_input(event.addr.0, event.value);
        inner_perm().permute_mut(&mut state);
        let sign = if event.is_receive { -event.multiplicity } else { event.multiplicity };
        for i in 0..14 {
            accumulator[i] += sign * state[i];
        }
    }
    accumulator
}

/// Derive compressed memory boundaries and the public accumulator for one event shard.
///
/// The record may then be proved with the normal per-shard Fiat-Shamir transcript. No shared
/// main-trace commitment phase is required.
pub fn prepare_event_shard_boundary(record: &mut ExecutionRecord<SP1Field>) -> [SP1Field; 14] {
    assert!(
        record.program.event_ranges.is_some(),
        "global memory boundaries are only defined for event-range programs"
    );
    assert!(
        record.commit_pv_hash_events.is_empty(),
        "event-shard descriptor public values currently require no CommitPublicValues event"
    );
    let balances = collect_record_memory_balances(record);
    let mut entries = balances.into_iter().collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(key, _)| *key);
    record.global_memory_boundary_events = entries
        .into_iter()
        .filter_map(|(key, balance)| {
            if balance.is_zero() {
                return None;
            }
            let canonical = balance.as_canonical_u32();
            let (multiplicity, is_receive) = if canonical <= SP1Field::ORDER_U32 / 2 {
                (balance, true)
            } else {
                (-balance, false)
            };
            Some(GlobalMemoryBoundaryEvent {
                addr: Address(SP1Field::from_canonical_u32(key.addr)),
                value: Block(key.value.map(SP1Field::from_canonical_u32)),
                multiplicity,
                is_receive,
            })
        })
        .collect();

    let accumulator = boundary_accumulator(&record.global_memory_boundary_events);
    record.public_values.global_cumulative_sum = SepticDigest(SepticCurve {
        x: SepticExtension(accumulator[..7].try_into().unwrap()),
        y: SepticExtension(accumulator[7..].try_into().unwrap()),
    });
    record.public_values.num_included_shard =
        SP1Field::from_canonical_usize(record.global_memory_boundary_events.len());
    record.public_values.digest =
        event_shard_program_digest(&record.program).expect("event shard digest is missing");
    accumulator
}

/// Prepare a complete in-memory batch and assert that its public accumulators close.
pub fn prepare_event_shard_boundaries(records: &mut [ExecutionRecord<SP1Field>]) {
    assert!(!records.is_empty(), "an event shard batch cannot be empty");
    let mut global_accumulator = [SP1Field::zero(); 14];
    for record in records {
        let accumulator = prepare_event_shard_boundary(record);
        global_accumulator.iter_mut().zip(accumulator).for_each(|(total, value)| *total += value);
    }

    assert!(
        global_accumulator.iter().all(Field::is_zero),
        "event-shard global memory accumulators do not close"
    );
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use slop_algebra::extension::BinomialExtensionField;
    use sp1_hypercube::inner_perm;
    use sp1_primitives::{SP1DiffusionMatrix, SP1Field};
    use sp1_recursion_executor::{
        instruction as instr, linear_program, Block, EventRange, Executor, MemAccessKind,
        RecursionEventRanges, D,
    };

    use crate::{machine::RecursionAir, test::run_test_recursion};

    use super::prepare_event_shard_boundaries;

    #[tokio::test]
    async fn independently_proves_two_memory_boundary_shards() {
        type A = RecursionAir<SP1Field, 3, 2>;

        let input = 0x3fc0;
        let output = 0x4010;
        let program = linear_program(vec![
            instr::mem(MemAccessKind::Write, 1, 0, input),
            instr::bf16_square(1, 1, 0),
            instr::mem(MemAccessKind::Read, 1, 1, output),
        ])
        .unwrap();
        let mut executor = Executor::<
            SP1Field,
            BinomialExtensionField<SP1Field, D>,
            SP1DiffusionMatrix,
        >::new(Arc::new(program.clone()), inner_perm());
        executor.witness_stream = Vec::<Block<SP1Field>>::new().into();
        executor.run().unwrap();

        let first_ranges = RecursionEventRanges {
            mem_const: EventRange { start: 0, end: 1 },
            ..RecursionEventRanges::default()
        };
        let second_ranges = RecursionEventRanges {
            mem_const: EventRange { start: 1, end: 2 },
            bf16_unary: EventRange { start: 0, end: 1 },
            ..RecursionEventRanges::default()
        };
        let parent_program =
            Arc::new(program.with_event_ranges(RecursionEventRanges::full(program.event_counts)));
        let parent_record = executor.record.into_event_shards(vec![parent_program]).pop().unwrap();
        let shard_programs = vec![
            Arc::new(program.with_event_ranges(first_ranges)),
            Arc::new(program.with_event_ranges(second_ranges)),
        ];
        // Exercise the adaptive planner's ability to split an already-sharded record again.
        let mut records = parent_record.into_event_shards(shard_programs);
        prepare_event_shard_boundaries(&mut records);

        assert_eq!(records[0].global_memory_boundary_events.len(), 1);
        assert_eq!(records[1].global_memory_boundary_events.len(), 1);
        for record in records {
            let shard_program = (*record.program).clone();
            run_test_recursion(vec![record], A::verillm_machine(), shard_program).await.unwrap();
        }
    }
}
