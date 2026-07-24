use std::{
    array,
    cell::UnsafeCell,
    mem::MaybeUninit,
    ops::{Add, AddAssign},
    sync::Arc,
};

use serde::{Deserialize, Serialize};
use slop_algebra::{AbstractField, Field, PrimeField32};
use sp1_hypercube::{air::SP1AirBuilder, InteractionKind, MachineRecord, PROOF_MAX_NUM_PVS};

use crate::{
    instruction::{HintBitsInstr, HintExt2FeltsInstr, HintInstr},
    public_values::RecursionPublicValues,
    Bf16AddSubEvent, Bf16DivEvent, Bf16MulEvent, Bf16UnaryEvent, ExtFeltEvent, Instruction,
    Poseidon2LinearLayerEvent, Poseidon2SBoxEvent, PrefixSumChecksEvent,
};

use super::{
    BaseAluEvent, CommitPublicValuesEvent, ExtAluEvent, MemEvent, Poseidon2Event, RecursionProgram,
    SelectEvent,
};

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct ExecutionRecord<F> {
    pub program: Arc<RecursionProgram<F>>,
    /// The index of the shard.
    pub index: u32,

    pub base_alu_events: Vec<BaseAluEvent<F>>,
    pub ext_alu_events: Vec<ExtAluEvent<F>>,
    pub mem_const_count: usize,
    pub mem_var_events: Vec<MemEvent<F>>,
    /// The public values.
    pub public_values: RecursionPublicValues<F>,

    pub ext_felt_conversion_events: Vec<ExtFeltEvent<F>>,
    pub poseidon2_events: Vec<Poseidon2Event<F>>,
    pub poseidon2_linear_layer_events: Vec<Poseidon2LinearLayerEvent<F>>,
    pub poseidon2_sbox_events: Vec<Poseidon2SBoxEvent<F>>,
    pub select_events: Vec<SelectEvent<F>>,
    pub bf16_mul_events: Vec<Bf16MulEvent<F>>,
    pub bf16_unary_events: Vec<Bf16UnaryEvent<F>>,
    pub bf16_div_events: Vec<Bf16DivEvent<F>>,
    pub bf16_add_sub_events: Vec<Bf16AddSubEvent<F>>,
    pub prefix_sum_checks_events: Vec<PrefixSumChecksEvent<F>>,
    pub commit_pv_hash_events: Vec<CommitPublicValuesEvent<F>>,
}

#[derive(Debug)]
pub struct UnsafeRecord<F> {
    pub base_alu_events: Vec<MaybeUninit<UnsafeCell<BaseAluEvent<F>>>>,
    pub ext_alu_events: Vec<MaybeUninit<UnsafeCell<ExtAluEvent<F>>>>,
    // Can be computed by the analysis step.
    pub mem_const_count: usize,
    pub mem_var_events: Vec<MaybeUninit<UnsafeCell<MemEvent<F>>>>,
    /// The public values.
    pub public_values: MaybeUninit<UnsafeCell<RecursionPublicValues<F>>>,

    pub ext_felt_conversion_events: Vec<MaybeUninit<UnsafeCell<ExtFeltEvent<F>>>>,
    pub poseidon2_events: Vec<MaybeUninit<UnsafeCell<Poseidon2Event<F>>>>,
    pub poseidon2_linear_layer_events: Vec<MaybeUninit<UnsafeCell<Poseidon2LinearLayerEvent<F>>>>,
    pub poseidon2_sbox_events: Vec<MaybeUninit<UnsafeCell<Poseidon2SBoxEvent<F>>>>,
    pub select_events: Vec<MaybeUninit<UnsafeCell<SelectEvent<F>>>>,
    pub bf16_mul_events: Vec<MaybeUninit<UnsafeCell<Bf16MulEvent<F>>>>,
    pub bf16_unary_events: Vec<MaybeUninit<UnsafeCell<Bf16UnaryEvent<F>>>>,
    pub bf16_div_events: Vec<MaybeUninit<UnsafeCell<Bf16DivEvent<F>>>>,
    pub bf16_add_sub_events: Vec<MaybeUninit<UnsafeCell<Bf16AddSubEvent<F>>>>,
    pub prefix_sum_checks_events: Vec<MaybeUninit<UnsafeCell<PrefixSumChecksEvent<F>>>>,
    pub commit_pv_hash_events: Vec<MaybeUninit<UnsafeCell<CommitPublicValuesEvent<F>>>>,
}

impl<F> UnsafeRecord<F> {
    /// # Safety
    ///
    /// The caller must ensure that the `UnsafeRecord` is fully initialized, this is
    /// done by the executor.
    pub unsafe fn into_record(
        self,
        program: Arc<RecursionProgram<F>>,
        index: u32,
    ) -> ExecutionRecord<F> {
        // SAFETY: `T` and `MaybeUninit<UnsafeCell<T>>` have the same memory layout.
        #[allow(clippy::missing_transmute_annotations)]
        ExecutionRecord {
            program,
            index,
            base_alu_events: std::mem::transmute(self.base_alu_events),
            ext_alu_events: std::mem::transmute(self.ext_alu_events),
            mem_const_count: self.mem_const_count,
            mem_var_events: std::mem::transmute(self.mem_var_events),
            public_values: self.public_values.assume_init().into_inner(),
            ext_felt_conversion_events: std::mem::transmute(self.ext_felt_conversion_events),
            poseidon2_events: std::mem::transmute(self.poseidon2_events),
            poseidon2_linear_layer_events: std::mem::transmute(self.poseidon2_linear_layer_events),
            poseidon2_sbox_events: std::mem::transmute(self.poseidon2_sbox_events),
            select_events: std::mem::transmute(self.select_events),
            bf16_mul_events: std::mem::transmute(self.bf16_mul_events),
            bf16_unary_events: std::mem::transmute(self.bf16_unary_events),
            bf16_div_events: std::mem::transmute(self.bf16_div_events),
            bf16_add_sub_events: std::mem::transmute(self.bf16_add_sub_events),
            prefix_sum_checks_events: std::mem::transmute(self.prefix_sum_checks_events),
            commit_pv_hash_events: std::mem::transmute(self.commit_pv_hash_events),
        }
    }

    pub fn new(event_counts: RecursionAirEventCount) -> Self
    where
        F: Field,
    {
        #[inline]
        fn create_uninit_vec<T>(len: usize) -> Vec<MaybeUninit<T>> {
            let mut vec = Vec::with_capacity(len);
            // SAFETY: The vector has enough capacity to hold the elements as we just allocated it,
            // and the type `T` is `MaybeUninit` which implies that an "uninitialized" value is OK.
            unsafe { vec.set_len(len) };
            vec
        }

        Self {
            base_alu_events: create_uninit_vec(event_counts.base_alu_events),
            ext_alu_events: create_uninit_vec(event_counts.ext_alu_events),
            mem_const_count: event_counts.mem_const_events,
            mem_var_events: create_uninit_vec(event_counts.mem_var_events),
            // Programs which do not expose explicit public values still produce a valid record.
            // A CommitPublicValues instruction overwrites this value during execution.
            public_values: MaybeUninit::new(UnsafeCell::new(RecursionPublicValues::default())),
            ext_felt_conversion_events: create_uninit_vec(event_counts.ext_felt_conversion_events),
            poseidon2_events: create_uninit_vec(event_counts.poseidon2_wide_events),
            poseidon2_linear_layer_events: create_uninit_vec(
                event_counts.poseidon2_linear_layer_events,
            ),
            poseidon2_sbox_events: create_uninit_vec(event_counts.poseidon2_sbox_events),
            select_events: create_uninit_vec(event_counts.select_events),
            bf16_mul_events: create_uninit_vec(event_counts.bf16_mul_events),
            bf16_unary_events: create_uninit_vec(event_counts.bf16_unary_events),
            bf16_div_events: create_uninit_vec(event_counts.bf16_div_events),
            bf16_add_sub_events: create_uninit_vec(event_counts.bf16_add_sub_events),
            prefix_sum_checks_events: create_uninit_vec(event_counts.prefix_sum_checks_events),
            commit_pv_hash_events: create_uninit_vec(event_counts.commit_pv_hash_events),
        }
    }
}

unsafe impl<F> Sync for UnsafeRecord<F> {}

impl<F: PrimeField32> MachineRecord for ExecutionRecord<F> {
    fn stats(&self) -> hashbrown::HashMap<String, usize> {
        [
            ("base_alu_events", self.base_alu_events.len()),
            ("ext_alu_events", self.ext_alu_events.len()),
            ("mem_const_count", self.mem_const_count),
            ("mem_var_events", self.mem_var_events.len()),
            ("ext_felt_conversion_events", self.ext_felt_conversion_events.len()),
            ("poseidon2_events", self.poseidon2_events.len()),
            ("poseidon2_linear_layer_events", self.poseidon2_linear_layer_events.len()),
            ("poseidon2_sbox_events", self.poseidon2_sbox_events.len()),
            ("select_events", self.select_events.len()),
            ("bf16_mul_events", self.bf16_mul_events.len()),
            ("bf16_unary_events", self.bf16_unary_events.len()),
            ("bf16_div_events", self.bf16_div_events.len()),
            ("bf16_add_sub_events", self.bf16_add_sub_events.len()),
            ("prefix_sum_checks_events", self.prefix_sum_checks_events.len()),
            ("commit_pv_hash_events", self.commit_pv_hash_events.len()),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect()
    }

    fn append(&mut self, other: &mut Self) {
        // Exhaustive destructuring for refactoring purposes.
        let Self {
            program: _,
            index: _,
            base_alu_events,
            ext_alu_events,
            mem_const_count,
            mem_var_events,
            public_values: _,
            ext_felt_conversion_events,
            poseidon2_events,
            poseidon2_linear_layer_events,
            poseidon2_sbox_events,
            select_events,
            bf16_mul_events,
            bf16_unary_events,
            bf16_div_events,
            bf16_add_sub_events,
            prefix_sum_checks_events,
            commit_pv_hash_events,
        } = self;
        base_alu_events.append(&mut other.base_alu_events);
        ext_alu_events.append(&mut other.ext_alu_events);
        *mem_const_count += other.mem_const_count;
        mem_var_events.append(&mut other.mem_var_events);
        ext_felt_conversion_events.append(&mut other.ext_felt_conversion_events);
        poseidon2_events.append(&mut other.poseidon2_events);
        poseidon2_linear_layer_events.append(&mut other.poseidon2_linear_layer_events);
        poseidon2_sbox_events.append(&mut other.poseidon2_sbox_events);
        select_events.append(&mut other.select_events);
        bf16_mul_events.append(&mut other.bf16_mul_events);
        bf16_unary_events.append(&mut other.bf16_unary_events);
        bf16_div_events.append(&mut other.bf16_div_events);
        bf16_add_sub_events.append(&mut other.bf16_add_sub_events);
        prefix_sum_checks_events.append(&mut other.prefix_sum_checks_events);
        commit_pv_hash_events.append(&mut other.commit_pv_hash_events);
    }

    fn public_values<T: AbstractField>(&self) -> Vec<T> {
        let pv_elms = self.public_values.as_array();

        let ret: [T; PROOF_MAX_NUM_PVS] = array::from_fn(|i| {
            if i < pv_elms.len() {
                T::from_canonical_u32(pv_elms[i].as_canonical_u32())
            } else {
                T::zero()
            }
        });

        ret.to_vec()
    }

    // No public value constraints for recursion public values.
    fn eval_public_values<AB: SP1AirBuilder>(_builder: &mut AB) {}

    fn interactions_in_public_values() -> Vec<InteractionKind> {
        vec![]
    }
}

impl<F: Copy> ExecutionRecord<F> {
    /// Split one globally executed record into trace shards without changing event order.
    ///
    /// Each supplied program must be a [`RecursionProgram::with_event_ranges`] view of this
    /// record's global program. The ranges must be contiguous and cover every supported event
    /// vector exactly once.
    #[must_use]
    pub fn into_event_shards(self, programs: Vec<Arc<RecursionProgram<F>>>) -> Vec<Self> {
        fn take_exact<T>(events: &mut impl Iterator<Item = T>, len: usize) -> Vec<T> {
            let result = events.take(len).collect::<Vec<_>>();
            assert_eq!(result.len(), len, "event shard range exceeds the global record");
            result
        }

        let Self {
            program: _,
            index: _,
            base_alu_events,
            ext_alu_events,
            mem_const_count,
            mem_var_events,
            public_values,
            ext_felt_conversion_events,
            poseidon2_events,
            poseidon2_linear_layer_events,
            poseidon2_sbox_events,
            select_events,
            bf16_mul_events,
            bf16_unary_events,
            bf16_div_events,
            bf16_add_sub_events,
            prefix_sum_checks_events,
            commit_pv_hash_events,
        } = self;
        assert!(ext_alu_events.is_empty());
        assert!(ext_felt_conversion_events.is_empty());
        assert!(poseidon2_linear_layer_events.is_empty());
        assert!(poseidon2_sbox_events.is_empty());
        assert!(select_events.is_empty());
        assert!(prefix_sum_checks_events.is_empty());

        let mut base_alu = base_alu_events.into_iter();
        let mut mem_var = mem_var_events.into_iter();
        let mut poseidon2 = poseidon2_events.into_iter();
        let mut bf16_mul = bf16_mul_events.into_iter();
        let mut bf16_unary = bf16_unary_events.into_iter();
        let mut bf16_div = bf16_div_events.into_iter();
        let mut bf16_add_sub = bf16_add_sub_events.into_iter();
        let mut commit_pv_hash = commit_pv_hash_events.into_iter();
        let mut expected = RecursionEventRanges::default();
        let mut consumed_mem_const = 0usize;

        let records = programs
            .into_iter()
            .enumerate()
            .map(|(index, program)| {
                let ranges = program
                    .event_ranges
                    .expect("an event shard program must contain explicit event ranges");
                assert_eq!(ranges.mem_const.start, expected.mem_const.end);
                assert_eq!(ranges.mem_var.start, expected.mem_var.end);
                assert_eq!(ranges.base_alu.start, expected.base_alu.end);
                assert_eq!(ranges.poseidon2_wide.start, expected.poseidon2_wide.end);
                assert_eq!(ranges.bf16_mul.start, expected.bf16_mul.end);
                assert_eq!(ranges.bf16_unary.start, expected.bf16_unary.end);
                assert_eq!(ranges.bf16_div.start, expected.bf16_div.end);
                assert_eq!(ranges.bf16_add_sub.start, expected.bf16_add_sub.end);
                assert_eq!(ranges.commit_pv_hash.start, expected.commit_pv_hash.end);
                expected = ranges;
                consumed_mem_const += ranges.mem_const.len();

                Self {
                    program,
                    index: u32::try_from(index).expect("too many recursion event shards"),
                    base_alu_events: take_exact(&mut base_alu, ranges.base_alu.len()),
                    ext_alu_events: Vec::new(),
                    mem_const_count: ranges.mem_const.len(),
                    mem_var_events: take_exact(&mut mem_var, ranges.mem_var.len()),
                    public_values,
                    ext_felt_conversion_events: Vec::new(),
                    poseidon2_events: take_exact(&mut poseidon2, ranges.poseidon2_wide.len()),
                    poseidon2_linear_layer_events: Vec::new(),
                    poseidon2_sbox_events: Vec::new(),
                    select_events: Vec::new(),
                    bf16_mul_events: take_exact(&mut bf16_mul, ranges.bf16_mul.len()),
                    bf16_unary_events: take_exact(&mut bf16_unary, ranges.bf16_unary.len()),
                    bf16_div_events: take_exact(&mut bf16_div, ranges.bf16_div.len()),
                    bf16_add_sub_events: take_exact(&mut bf16_add_sub, ranges.bf16_add_sub.len()),
                    prefix_sum_checks_events: Vec::new(),
                    commit_pv_hash_events: take_exact(
                        &mut commit_pv_hash,
                        ranges.commit_pv_hash.len(),
                    ),
                }
            })
            .collect::<Vec<_>>();

        assert_eq!(consumed_mem_const, mem_const_count);
        assert!(base_alu.next().is_none());
        assert!(mem_var.next().is_none());
        assert!(poseidon2.next().is_none());
        assert!(bf16_mul.next().is_none());
        assert!(bf16_unary.next().is_none());
        assert!(bf16_div.next().is_none());
        assert!(bf16_add_sub.next().is_none());
        assert!(commit_pv_hash.next().is_none());
        records
    }
}

impl<F: Field> ExecutionRecord<F> {
    pub fn compute_event_counts<'a>(
        instrs: impl Iterator<Item = &'a Instruction<F>> + 'a,
    ) -> RecursionAirEventCount {
        instrs.fold(RecursionAirEventCount::default(), Add::add)
    }
}

/// A half-open range in one global recursion event vector.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct EventRange {
    pub start: usize,
    pub end: usize,
}

impl EventRange {
    #[must_use]
    pub const fn len(self) -> usize {
        self.end - self.start
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// Active slices of the global event vectors for a trace shard.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct RecursionEventRanges {
    pub mem_const: EventRange,
    pub mem_var: EventRange,
    pub base_alu: EventRange,
    pub poseidon2_wide: EventRange,
    pub bf16_mul: EventRange,
    pub bf16_unary: EventRange,
    pub bf16_div: EventRange,
    pub bf16_add_sub: EventRange,
    pub commit_pv_hash: EventRange,
}

impl RecursionEventRanges {
    #[must_use]
    pub const fn event_counts(self) -> RecursionAirEventCount {
        RecursionAirEventCount {
            mem_const_events: self.mem_const.len(),
            mem_var_events: self.mem_var.len(),
            base_alu_events: self.base_alu.len(),
            ext_alu_events: 0,
            ext_felt_conversion_events: 0,
            poseidon2_wide_events: self.poseidon2_wide.len(),
            poseidon2_linear_layer_events: 0,
            poseidon2_sbox_events: 0,
            select_events: 0,
            bf16_mul_events: self.bf16_mul.len(),
            bf16_unary_events: self.bf16_unary.len(),
            bf16_div_events: self.bf16_div.len(),
            bf16_add_sub_events: self.bf16_add_sub.len(),
            prefix_sum_checks_events: 0,
            commit_pv_hash_events: self.commit_pv_hash.len(),
        }
    }

    #[must_use]
    pub const fn full(counts: RecursionAirEventCount) -> Self {
        Self {
            mem_const: EventRange { start: 0, end: counts.mem_const_events },
            mem_var: EventRange { start: 0, end: counts.mem_var_events },
            base_alu: EventRange { start: 0, end: counts.base_alu_events },
            poseidon2_wide: EventRange { start: 0, end: counts.poseidon2_wide_events },
            bf16_mul: EventRange { start: 0, end: counts.bf16_mul_events },
            bf16_unary: EventRange { start: 0, end: counts.bf16_unary_events },
            bf16_div: EventRange { start: 0, end: counts.bf16_div_events },
            bf16_add_sub: EventRange { start: 0, end: counts.bf16_add_sub_events },
            commit_pv_hash: EventRange { start: 0, end: counts.commit_pv_hash_events },
        }
    }
}

#[derive(Default, Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct RecursionAirEventCount {
    pub mem_const_events: usize,
    pub mem_var_events: usize,
    pub base_alu_events: usize,
    pub ext_alu_events: usize,
    pub ext_felt_conversion_events: usize,
    pub poseidon2_wide_events: usize,
    pub poseidon2_linear_layer_events: usize,
    pub poseidon2_sbox_events: usize,
    pub select_events: usize,
    pub bf16_mul_events: usize,
    pub bf16_unary_events: usize,
    pub bf16_div_events: usize,
    pub bf16_add_sub_events: usize,
    pub prefix_sum_checks_events: usize,
    pub commit_pv_hash_events: usize,
}

impl<F> AddAssign<&Instruction<F>> for RecursionAirEventCount {
    #[inline]
    fn add_assign(&mut self, rhs: &Instruction<F>) {
        match rhs {
            Instruction::BaseAlu(_) => self.base_alu_events += 1,
            Instruction::ExtAlu(_) => self.ext_alu_events += 1,
            Instruction::ExtFelt(_) => self.ext_felt_conversion_events += 1,
            Instruction::Mem(_) => self.mem_const_events += 1,
            Instruction::Poseidon2(_) => self.poseidon2_wide_events += 1,
            Instruction::Poseidon2LinearLayer(_) => self.poseidon2_linear_layer_events += 1,
            Instruction::Poseidon2SBox(_) => self.poseidon2_sbox_events += 1,
            Instruction::Select(_) => self.select_events += 1,
            Instruction::Bf16Mul(_) => self.bf16_mul_events += 1,
            Instruction::Bf16Unary(_) => self.bf16_unary_events += 1,
            Instruction::Bf16Div(_) => self.bf16_div_events += 1,
            Instruction::Bf16AddSub(_) => self.bf16_add_sub_events += 1,
            Instruction::Bf16LinearBatch(instruction) => {
                self.bf16_mul_events += instruction.mul_count();
                self.bf16_add_sub_events += instruction.add_sub_count();
            }
            Instruction::Bf16MeanBatch(instruction) => {
                self.bf16_add_sub_events += instruction.add_sub_count();
                self.bf16_div_events += 1;
            }
            Instruction::Hint(HintInstr { output_addrs_mults })
            | Instruction::HintBits(HintBitsInstr {
                output_addrs_mults,
                input_addr: _, // No receive interaction for the hint operation
            }) => self.mem_var_events += output_addrs_mults.len(),
            Instruction::HintExt2Felts(HintExt2FeltsInstr {
                output_addrs_mults,
                input_addr: _, // No receive interaction for the hint operation
            }) => self.mem_var_events += output_addrs_mults.len(),
            Instruction::PrefixSumChecks(instr) => {
                self.prefix_sum_checks_events += instr.addrs.x1.len()
            }
            Instruction::HintAddCurve(instr) => {
                self.mem_var_events += instr.output_x_addrs_mults.len();
                self.mem_var_events += instr.output_y_addrs_mults.len();
            }
            Instruction::CommitPublicValues(_) => self.commit_pv_hash_events += 1,
            Instruction::Print(_) | Instruction::DebugBacktrace(_) => {}
        }
    }
}

impl<F> Add<&Instruction<F>> for RecursionAirEventCount {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: &Instruction<F>) -> Self::Output {
        self += rhs;
        self
    }
}
