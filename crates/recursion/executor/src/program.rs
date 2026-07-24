use crate::{analyzed::AnalyzedInstruction, shape::RecursionShape, *};
use serde::{Deserialize, Serialize};
use slop_algebra::{AbstractField, Field};
use slop_symmetric::Permutation;
use sp1_hypercube::{
    air::MachineProgram, inner_perm, septic_digest::SepticDigest, UntrustedConfig,
};
use sp1_primitives::SP1Field;
use std::ops::{Deref, DerefMut};

pub use basic_block::BasicBlock;
pub use raw::RawProgram;
pub use seq_block::SeqBlock;

/// Domain and version for the descriptor committed by an event shard.
pub const EVENT_SHARD_DESCRIPTOR_DOMAIN: u32 = 0x564c_4d53;
pub const EVENT_SHARD_DESCRIPTOR_VERSION: u32 = 1;

/// Commit the global event counts and this shard's exact ranges.
///
/// The digest is constrained by the public-values chip from preprocessed program metadata. A
/// recursion join can therefore check range coverage without trusting a host manifest.
#[must_use]
pub fn event_shard_descriptor_digest(
    global_counts: RecursionAirEventCount,
    ranges: RecursionEventRanges,
) -> [SP1Field; DIGEST_SIZE] {
    let counts = [
        global_counts.mem_const_events,
        global_counts.mem_var_events,
        global_counts.base_alu_events,
        global_counts.poseidon2_wide_events,
        global_counts.bf16_mul_events,
        global_counts.bf16_unary_events,
        global_counts.bf16_div_events,
        global_counts.bf16_add_sub_events,
        global_counts.commit_pv_hash_events,
    ];
    let ranges = [
        ranges.mem_const,
        ranges.mem_var,
        ranges.base_alu,
        ranges.poseidon2_wide,
        ranges.bf16_mul,
        ranges.bf16_unary,
        ranges.bf16_div,
        ranges.bf16_add_sub,
        ranges.commit_pv_hash,
    ];
    let mut fields = Vec::with_capacity(2 + counts.len() + 2 * ranges.len());
    fields.push(SP1Field::from_canonical_u32(EVENT_SHARD_DESCRIPTOR_VERSION));
    fields.extend(counts.map(SP1Field::from_canonical_usize));
    for range in ranges {
        fields.push(SP1Field::from_canonical_usize(range.start));
        fields.push(SP1Field::from_canonical_usize(range.end));
    }

    let mut state = [SP1Field::zero(); PERMUTATION_WIDTH];
    state[0] = SP1Field::from_canonical_u32(EVENT_SHARD_DESCRIPTOR_DOMAIN);
    state[1] = SP1Field::from_canonical_usize(fields.len());
    inner_perm().permute_mut(&mut state);
    for chunk in fields.chunks(HASH_RATE) {
        state[..HASH_RATE].fill(SP1Field::zero());
        state[..chunk.len()].copy_from_slice(chunk);
        inner_perm().permute_mut(&mut state);
    }
    state[..DIGEST_SIZE].try_into().unwrap()
}

#[must_use]
pub fn event_shard_program_digest(
    program: &RecursionProgram<SP1Field>,
) -> Option<[SP1Field; DIGEST_SIZE]> {
    let ranges = program.event_ranges?;
    let global_counts = program
        .global_event_counts
        .expect("an event-range program must retain its global event counts");
    Some(event_shard_descriptor_digest(global_counts, ranges))
}

/// A well-formed recursion program. See [`Self::new_unchecked`] for guaranteed (safety) invariants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(transparent)]
pub struct RecursionProgram<F>(RootProgram<F>);

impl<F> RecursionProgram<F> {
    /// # Safety
    /// The given program must be well formed. This is defined as the following:
    /// - reads are performed after writes, according to a "happens-before" relation; and
    /// - an address is written to at most once.
    ///
    /// The "happens-before" relation is defined as follows:
    /// - It is a strict partial order, meaning it is transitive, irreflexive, and asymmetric.
    /// - Instructions in a `BasicBlock` are linearly ordered.
    /// - `SeqBlock`s in a `RawProgram` are linearly ordered, meaning:
    ///     - Each `SeqBlock` has a set of initial instructions `I` and final instructions `O`.
    ///     - For `SeqBlock::Basic`:
    ///         - `I` is the singleton consisting of the first instruction in the enclosed
    ///           `BasicBlock`.
    ///         - `O` is the singleton consisting of the last instruction in the enclosed
    ///           `BasicBlock`.
    ///     - For `SeqBlock::Parallel`:
    ///         - `I` is the set of initial instructions `I` in the first `SeqBlock` of the enclosed
    ///           `RawProgram`.
    ///         - `O` is the set of final instructions in the last `SeqBlock` of the enclosed
    ///           `RawProgram`.
    ///     - For consecutive `SeqBlock`s, each element of the first one's `O` happens before the
    ///       second one's `I`.
    ///
    /// - The last condition is the event count analysis is done correctly see [`crate::analyzed`].
    pub unsafe fn new_unchecked(program: RootProgram<F>) -> Self {
        Self(program)
    }

    pub fn into_inner(self) -> RootProgram<F> {
        self.0
    }

    /// Return the global event slices represented by this program.
    #[must_use]
    pub fn event_ranges(&self) -> RecursionEventRanges {
        self.event_ranges.unwrap_or_else(|| RecursionEventRanges::full(self.event_counts))
    }
}

impl<F: Clone> RecursionProgram<F> {
    /// Clone this global program as one event-trace shard.
    ///
    /// The instruction graph and global address space stay unchanged. Chips use `ranges` to
    /// select matching preprocessed access rows, while the shard record contains the same slices
    /// of the main event vectors.
    #[must_use]
    pub fn with_event_ranges(&self, ranges: RecursionEventRanges) -> Self {
        let full = self.event_ranges();
        let contained = |range: EventRange, outer: EventRange| {
            outer.start <= range.start && range.end <= outer.end
        };
        assert!(contained(ranges.mem_const, full.mem_const));
        assert!(contained(ranges.mem_var, full.mem_var));
        assert!(contained(ranges.base_alu, full.base_alu));
        assert!(contained(ranges.poseidon2_wide, full.poseidon2_wide));
        assert!(contained(ranges.bf16_mul, full.bf16_mul));
        assert!(contained(ranges.bf16_unary, full.bf16_unary));
        assert!(contained(ranges.bf16_div, full.bf16_div));
        assert!(contained(ranges.bf16_add_sub, full.bf16_add_sub));
        assert!(contained(ranges.commit_pv_hash, full.commit_pv_hash));

        let mut root = self.0.clone();
        root.global_event_counts = Some(root.global_event_counts.unwrap_or(root.event_counts));
        root.event_counts = ranges.event_counts();
        root.event_ranges = Some(ranges);
        // SAFETY: only event summary metadata changed; the instruction graph remains the already
        // validated global program.
        unsafe { Self::new_unchecked(root) }
    }
}

impl<F> Default for RecursionProgram<F> {
    fn default() -> Self {
        // SAFETY: An empty program is always well formed.
        unsafe { Self::new_unchecked(RootProgram::default()) }
    }
}

impl<F> Deref for RecursionProgram<F> {
    type Target = RootProgram<F>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F> DerefMut for RecursionProgram<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F: Field> MachineProgram<F> for RecursionProgram<F> {
    fn pc_start(&self) -> [F; 3] {
        [F::zero(), F::zero(), F::zero()]
    }

    fn initial_global_cumulative_sum(&self) -> SepticDigest<F> {
        SepticDigest::<F>::zero()
    }

    fn untrusted_config(&self) -> UntrustedConfig<F> {
        UntrustedConfig::zero()
    }
}

#[cfg(any(test, feature = "program_validation"))]
pub use validation::*;

#[cfg(any(test, feature = "program_validation"))]
mod validation {
    use super::*;

    use std::{fmt::Debug, iter, mem};

    use range_set_blaze::{MultiwayRangeSetBlazeRef, RangeSetBlaze};
    use slop_algebra::PrimeField32;
    use smallvec::{smallvec, SmallVec};
    use thiserror::Error;

    #[derive(Error, Debug)]
    pub enum StructureError<F: Debug> {
        #[error("tried to read from uninitialized address {addr:?}. instruction: {instr:?}")]
        ReadFromUninit { addr: Address<F>, instr: Instruction<F> },
    }

    #[derive(Error, Debug)]
    pub enum SummaryError {
        #[error("`total_memory` is insufficient. configured: {configured}. required: {required}")]
        OutOfMemory { configured: usize, required: usize },
    }

    #[derive(Error, Debug)]
    pub enum ValidationError<F: Debug> {
        Structure(#[from] StructureError<F>),
        Summary(#[from] SummaryError),
    }

    impl<F: PrimeField32> RecursionProgram<F> {
        /// Validate the program without modifying its summary metadata.
        pub fn try_new_unmodified(
            program: RootProgram<F>,
        ) -> Result<Self, Box<ValidationError<F>>> {
            let written_addrs = try_written_addrs(smallvec![], &program.inner)
                .map_err(|e| ValidationError::from(*e))?;
            if let Some(required) = written_addrs.last().map(|x| x as usize + 1) {
                let configured = program.total_memory;
                if required > configured {
                    Err(Box::new(SummaryError::OutOfMemory { configured, required }.into()))?
                }
            }
            // SAFETY: We just checked all the invariants.
            Ok(unsafe { Self::new_unchecked(program) })
        }

        /// Validate the program, modifying summary metadata if necessary.
        pub fn try_new(mut program: RootProgram<F>) -> Result<Self, Box<StructureError<F>>> {
            let written_addrs = try_written_addrs(smallvec![], &program.inner)?;
            program.total_memory = written_addrs.last().map(|x| x as usize + 1).unwrap_or_default();
            // SAFETY: We just checked/enforced all the invariants.
            Ok(unsafe { Self::new_unchecked(program) })
        }
    }

    fn try_written_addrs<F: PrimeField32>(
        readable_addrs: SmallVec<[&RangeSetBlaze<u32>; 3]>,
        program: &RawProgram<AnalyzedInstruction<F>>,
    ) -> Result<RangeSetBlaze<u32>, Box<StructureError<F>>> {
        let mut written_addrs = RangeSetBlaze::<u32>::new();
        for block in &program.seq_blocks {
            match block {
                SeqBlock::Basic(basic_block) => {
                    for instr in &basic_block.instrs {
                        let (inputs, outputs) = instr.inner.io_addrs();
                        inputs.into_iter().try_for_each(|i| {
                            let i_u32 = i.0.as_canonical_u32();
                            iter::once(&written_addrs)
                                .chain(readable_addrs.iter().copied())
                                .any(|s| s.contains(i_u32))
                                .then_some(())
                                .ok_or_else(|| {
                                    Box::new(StructureError::ReadFromUninit {
                                        addr: i,
                                        instr: instr.inner.clone(),
                                    })
                                })
                        })?;
                        written_addrs.extend(outputs.iter().map(|o| o.0.as_canonical_u32()));
                    }
                }
                SeqBlock::Parallel(programs) => {
                    let par_written_addrs = programs
                        .iter()
                        .map(|subprogram| {
                            let sub_readable_addrs = iter::once(&written_addrs)
                                .chain(readable_addrs.iter().copied())
                                .collect();

                            try_written_addrs(sub_readable_addrs, subprogram)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    written_addrs =
                        iter::once(mem::take(&mut written_addrs)).chain(par_written_addrs).union();
                }
            }
        }
        Ok(written_addrs)
    }

    impl<F: PrimeField32> RootProgram<F> {
        pub fn validate(self) -> Result<RecursionProgram<F>, Box<StructureError<F>>> {
            RecursionProgram::try_new(self)
        }
    }

    pub fn linear_program<F: PrimeField32>(
        instrs: Vec<Instruction<F>>,
    ) -> Result<RecursionProgram<F>, Box<StructureError<F>>> {
        let (analyzed, counts) =
            RawProgram { seq_blocks: vec![SeqBlock::Basic(BasicBlock { instrs })] }.analyze();

        RootProgram {
            inner: analyzed,
            total_memory: 0,
            shape: None,
            event_counts: counts,
            global_event_counts: None,
            event_ranges: None,
        }
        .validate()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootProgram<F> {
    pub inner: raw::RawProgram<AnalyzedInstruction<F>>,
    pub total_memory: usize,
    pub shape: Option<RecursionShape<F>>,
    pub event_counts: RecursionAirEventCount,
    /// Event counts of the unsplit program. Present on an event-range view.
    #[serde(default)]
    pub global_event_counts: Option<RecursionAirEventCount>,
    /// Optional slices of the global event vectors used by one trace shard.
    pub event_ranges: Option<RecursionEventRanges>,
}

// `Default` without bounds on the type parameter.
impl<F> Default for RootProgram<F> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
            total_memory: Default::default(),
            shape: None,
            event_counts: Default::default(),
            global_event_counts: None,
            event_ranges: None,
        }
    }
}

pub mod raw {
    use std::iter::Flatten;

    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RawProgram<T> {
        pub seq_blocks: Vec<SeqBlock<T>>,
    }

    // `Default` without bounds on the type parameter.
    impl<T> Default for RawProgram<T> {
        fn default() -> Self {
            Self { seq_blocks: Default::default() }
        }
    }

    impl<T> RawProgram<T> {
        pub fn iter(&self) -> impl Iterator<Item = &'_ T> {
            self.seq_blocks.iter().flatten()
        }
        pub fn iter_mut(&mut self) -> impl Iterator<Item = &'_ mut T> {
            self.seq_blocks.iter_mut().flatten()
        }
    }

    impl<T> IntoIterator for RawProgram<T> {
        type Item = T;

        type IntoIter = Flatten<<Vec<SeqBlock<T>> as IntoIterator>::IntoIter>;

        fn into_iter(self) -> Self::IntoIter {
            self.seq_blocks.into_iter().flatten()
        }
    }

    impl<'a, T> IntoIterator for &'a RawProgram<T> {
        type Item = &'a T;

        type IntoIter = Flatten<<&'a Vec<SeqBlock<T>> as IntoIterator>::IntoIter>;

        fn into_iter(self) -> Self::IntoIter {
            self.seq_blocks.iter().flatten()
        }
    }

    impl<'a, T> IntoIterator for &'a mut RawProgram<T> {
        type Item = &'a mut T;

        type IntoIter = Flatten<<&'a mut Vec<SeqBlock<T>> as IntoIterator>::IntoIter>;

        fn into_iter(self) -> Self::IntoIter {
            self.seq_blocks.iter_mut().flatten()
        }
    }
}

pub mod seq_block {
    use std::iter::Flatten;

    use super::*;

    /// Segments that may be sequentially composed.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum SeqBlock<T> {
        /// One basic block.
        Basic(BasicBlock<T>),
        /// Many blocks to be run in parallel.
        Parallel(Vec<RawProgram<T>>),
    }

    impl<T> SeqBlock<T> {
        pub fn iter(&self) -> Iter<'_, T> {
            self.into_iter()
        }

        pub fn iter_mut(&mut self) -> IterMut<'_, T> {
            self.into_iter()
        }
    }

    // Bunch of iterator boilerplate.
    #[derive(Debug)]
    pub enum Iter<'a, T> {
        Basic(<&'a Vec<T> as IntoIterator>::IntoIter),
        Parallel(Box<Flatten<<&'a Vec<RawProgram<T>> as IntoIterator>::IntoIter>>),
    }

    impl<'a, T> Iterator for Iter<'a, T> {
        type Item = &'a T;

        fn next(&mut self) -> Option<Self::Item> {
            match self {
                Iter::Basic(it) => it.next(),
                Iter::Parallel(it) => it.next(),
            }
        }
    }

    impl<'a, T> IntoIterator for &'a SeqBlock<T> {
        type Item = &'a T;

        type IntoIter = Iter<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            match self {
                SeqBlock::Basic(basic_block) => Iter::Basic(basic_block.instrs.iter()),
                SeqBlock::Parallel(vec) => Iter::Parallel(Box::new(vec.iter().flatten())),
            }
        }
    }

    #[derive(Debug)]
    pub enum IterMut<'a, T> {
        Basic(<&'a mut Vec<T> as IntoIterator>::IntoIter),
        Parallel(Box<Flatten<<&'a mut Vec<RawProgram<T>> as IntoIterator>::IntoIter>>),
    }

    impl<'a, T> Iterator for IterMut<'a, T> {
        type Item = &'a mut T;

        fn next(&mut self) -> Option<Self::Item> {
            match self {
                IterMut::Basic(it) => it.next(),
                IterMut::Parallel(it) => it.next(),
            }
        }
    }

    impl<'a, T> IntoIterator for &'a mut SeqBlock<T> {
        type Item = &'a mut T;

        type IntoIter = IterMut<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            match self {
                SeqBlock::Basic(basic_block) => IterMut::Basic(basic_block.instrs.iter_mut()),
                SeqBlock::Parallel(vec) => IterMut::Parallel(Box::new(vec.iter_mut().flatten())),
            }
        }
    }

    #[derive(Debug, Clone)]
    pub enum IntoIter<T> {
        Basic(<Vec<T> as IntoIterator>::IntoIter),
        Parallel(Box<Flatten<<Vec<RawProgram<T>> as IntoIterator>::IntoIter>>),
    }

    impl<T> Iterator for IntoIter<T> {
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            match self {
                IntoIter::Basic(it) => it.next(),
                IntoIter::Parallel(it) => it.next(),
            }
        }
    }

    impl<T> IntoIterator for SeqBlock<T> {
        type Item = T;

        type IntoIter = IntoIter<T>;

        fn into_iter(self) -> Self::IntoIter {
            match self {
                SeqBlock::Basic(basic_block) => IntoIter::Basic(basic_block.instrs.into_iter()),
                SeqBlock::Parallel(vec) => IntoIter::Parallel(Box::new(vec.into_iter().flatten())),
            }
        }
    }
}

pub mod basic_block {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BasicBlock<T> {
        pub instrs: Vec<T>,
    }

    // Less restrictive trait bounds.
    impl<T> Default for BasicBlock<T> {
        fn default() -> Self {
            Self { instrs: Default::default() }
        }
    }
}
