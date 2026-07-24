//! Shape-aware shard planning for the VeriLLM BF16 traces.
//!
//! A planner unit is deliberately a semantic unit supplied by the caller (for example, one
//! complete output column or one complete attention head). The planner never cuts inside that
//! unit. It packs as many equal contiguous units as possible while respecting every BF16 chip's
//! row limit and a conservative combined main/preprocessed trace-area limit.

use crate::chips::bf16::{
    BF16_ADD_SUB_COLS, BF16_ADD_SUB_PREPROCESSED_COLS, BF16_DIV_COLS, BF16_DIV_PREPROCESSED_COLS,
    BF16_LOOKUP_COLS, BF16_LOOKUP_PREPROCESSED_COLS, BF16_LOOKUP_ROWS, BF16_MUL_COLS,
    BF16_MUL_PREPROCESSED_COLS, BF16_UNARY_COLS, BF16_UNARY_PREPROCESSED_COLS,
    NUM_BF16_ADD_SUB_EVENTS_PER_ROW, NUM_BF16_MUL_EVENTS_PER_ROW,
};
use crate::chips::{
    alu_base::{NUM_BASE_ALU_COLS, NUM_BASE_ALU_ENTRIES_PER_ROW, NUM_BASE_ALU_PREPROCESSED_COLS},
    global_memory_boundary::GLOBAL_MEMORY_BOUNDARY_COLS,
    mem::{
        constant::{
            NUM_CONST_MEM_ENTRIES_PER_ROW, NUM_MEM_INIT_COLS as NUM_MEM_CONST_COLS,
            NUM_MEM_PREPROCESSED_INIT_COLS as NUM_MEM_CONST_PREPROCESSED_COLS,
        },
        variable::{
            NUM_MEM_INIT_COLS as NUM_MEM_VAR_COLS,
            NUM_MEM_PREPROCESSED_INIT_COLS as NUM_MEM_VAR_PREPROCESSED_COLS,
        },
    },
    poseidon2_wide::columns::preprocessed::Poseidon2PreprocessedColsWide,
    public_values::{
        NUM_PUBLIC_VALUES_COLS, NUM_PUBLIC_VALUES_PREPROCESSED_COLS, PUB_VALUES_LOG_HEIGHT,
    },
};
use sp1_hypercube::operations::poseidon2::permutation::Poseidon2Degree3Cols;
use sp1_recursion_executor::RecursionAirEventCount;
pub use sp1_recursion_executor::{EventRange, RecursionEventRanges};

/// The default maximum height used by the full-size VeriLLM leaf proofs.
pub const DEFAULT_MAX_LOG_ROWS: usize = 19;

/// The Core-style upper bound used for the combined main and preprocessed trace area.
pub const DEFAULT_MAX_TRACE_AREA: usize = (1 << 28) + (1 << 27);

/// Core's default maximum per-chip height for execution shards.
pub const DEFAULT_GLOBAL_MAX_LOG_ROWS: usize = 22;

/// Space reserved for Memory, Poseidon2, the BF16 lookup table, and other non-arithmetic chips.
///
/// The arithmetic planner cannot know these rows without compiling the complete circuit. Full
/// proofs enforce [`DEFAULT_MAX_TRACE_AREA`] again against the exact proof shape before proving.
pub const DEFAULT_NON_ARITHMETIC_AREA_RESERVE: usize = 1 << 25;

/// BF16 events emitted by one indivisible semantic unit.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Bf16EventsPerUnit {
    pub mul: usize,
    pub add_sub: usize,
    pub unary: usize,
    pub div: usize,
}

/// Limits used while packing natural units into one leaf proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShardLimits {
    pub max_log_rows: usize,
    pub max_trace_area: usize,
    pub non_arithmetic_area_reserve: usize,
}

impl ShardLimits {
    #[must_use]
    pub const fn full() -> Self {
        Self {
            max_log_rows: DEFAULT_MAX_LOG_ROWS,
            max_trace_area: DEFAULT_MAX_TRACE_AREA,
            non_arithmetic_area_reserve: DEFAULT_NON_ARITHMETIC_AREA_RESERVE,
        }
    }

    /// Limits for the global event-stream sharding path.
    #[must_use]
    pub const fn core_style() -> Self {
        Self {
            max_log_rows: DEFAULT_GLOBAL_MAX_LOG_ROWS,
            max_trace_area: DEFAULT_MAX_TRACE_AREA,
            // The global estimator below accounts for every chip in `verillm_machine`.
            non_arithmetic_area_reserve: 0,
        }
    }

    #[must_use]
    pub const fn max_rows(self) -> usize {
        1 << self.max_log_rows
    }
}

/// Exact padded row and area estimate for one `verillm_machine` shard.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VerillmTraceEstimate {
    pub memory_const_rows: usize,
    pub memory_var_rows: usize,
    pub base_alu_rows: usize,
    pub poseidon2_rows: usize,
    pub bf16_lookup_rows: usize,
    pub bf16_mul_rows: usize,
    pub bf16_unary_rows: usize,
    pub bf16_div_rows: usize,
    pub bf16_add_sub_rows: usize,
    /// Minimum rows; the exact compressed boundary count is only known after execution.
    pub global_memory_boundary_rows: usize,
    pub public_values_rows: usize,
    pub preprocessed_area: usize,
    pub main_area: usize,
    pub total_area: usize,
}

impl VerillmTraceEstimate {
    #[must_use]
    pub fn max_rows(self) -> usize {
        [
            self.memory_const_rows,
            self.memory_var_rows,
            self.base_alu_rows,
            self.poseidon2_rows,
            self.bf16_lookup_rows,
            self.bf16_mul_rows,
            self.bf16_unary_rows,
            self.bf16_div_rows,
            self.bf16_add_sub_rows,
            self.global_memory_boundary_rows,
            self.public_values_rows,
        ]
        .into_iter()
        .max()
        .unwrap_or_default()
    }

    #[must_use]
    pub fn fits(self, limits: ShardLimits) -> bool {
        self.max_rows() <= limits.max_rows() && self.total_area <= limits.max_trace_area
    }
}

/// One balanced slice of every global event vector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventShard {
    pub index: usize,
    pub ranges: RecursionEventRanges,
    pub estimate: VerillmTraceEstimate,
}

/// A Core-style event/trace-area plan for one global recursion execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EventShardPlan {
    pub global_counts: RecursionAirEventCount,
    pub limits: ShardLimits,
    pub shards: Vec<EventShard>,
}

const VERILLM_VAR_EVENTS_PER_ROW: usize = 2;
const LOG_STACKING_HEIGHT: usize = 22;

fn proof_rows(events: usize, lanes: usize) -> usize {
    events.div_ceil(lanes).next_multiple_of(32).max(16)
}

/// Estimate the exact proof-shape area used by `RecursionAir<_, 3, 2>::verillm_machine`.
#[must_use]
pub fn estimate_verillm_trace(counts: RecursionAirEventCount) -> VerillmTraceEstimate {
    assert_eq!(counts.ext_alu_events, 0, "ExtAlu is not in verillm_machine");
    assert_eq!(counts.ext_felt_conversion_events, 0, "ExtFeltConvert is not in verillm_machine");
    assert_eq!(
        counts.poseidon2_linear_layer_events, 0,
        "Poseidon2LinearLayer is not in verillm_machine"
    );
    assert_eq!(counts.poseidon2_sbox_events, 0, "Poseidon2SBox is not in verillm_machine");
    assert_eq!(counts.select_events, 0, "Select is not in verillm_machine");
    assert_eq!(counts.prefix_sum_checks_events, 0, "PrefixSumChecks is not in verillm_machine");

    let memory_const_rows = proof_rows(counts.mem_const_events, NUM_CONST_MEM_ENTRIES_PER_ROW);
    let memory_var_rows = proof_rows(counts.mem_var_events, VERILLM_VAR_EVENTS_PER_ROW);
    let base_alu_rows = proof_rows(counts.base_alu_events, NUM_BASE_ALU_ENTRIES_PER_ROW);
    let poseidon2_rows = proof_rows(counts.poseidon2_wide_events, 1);
    let bf16_lookup_rows = BF16_LOOKUP_ROWS;
    let bf16_mul_rows = proof_rows(counts.bf16_mul_events, NUM_BF16_MUL_EVENTS_PER_ROW);
    let bf16_unary_rows = proof_rows(counts.bf16_unary_events, 1);
    let bf16_div_rows = proof_rows(counts.bf16_div_events, 1);
    let bf16_add_sub_rows = proof_rows(counts.bf16_add_sub_events, NUM_BF16_ADD_SUB_EVENTS_PER_ROW);
    let global_memory_boundary_rows = 16;
    let public_values_rows = 1 << PUB_VALUES_LOG_HEIGHT;

    let preprocessed_area = memory_const_rows * NUM_MEM_CONST_PREPROCESSED_COLS
        + memory_var_rows * NUM_MEM_VAR_PREPROCESSED_COLS * VERILLM_VAR_EVENTS_PER_ROW
        + base_alu_rows * NUM_BASE_ALU_PREPROCESSED_COLS
        + poseidon2_rows * core::mem::size_of::<Poseidon2PreprocessedColsWide<u8>>()
        + bf16_lookup_rows * BF16_LOOKUP_PREPROCESSED_COLS
        + bf16_mul_rows * BF16_MUL_PREPROCESSED_COLS
        + bf16_unary_rows * BF16_UNARY_PREPROCESSED_COLS
        + bf16_div_rows * BF16_DIV_PREPROCESSED_COLS
        + bf16_add_sub_rows * BF16_ADD_SUB_PREPROCESSED_COLS
        + public_values_rows * NUM_PUBLIC_VALUES_PREPROCESSED_COLS;
    let main_area = memory_const_rows * NUM_MEM_CONST_COLS
        + memory_var_rows * NUM_MEM_VAR_COLS * VERILLM_VAR_EVENTS_PER_ROW
        + base_alu_rows * NUM_BASE_ALU_COLS
        + poseidon2_rows * core::mem::size_of::<Poseidon2Degree3Cols<u8>>()
        + bf16_lookup_rows * BF16_LOOKUP_COLS
        + bf16_mul_rows * BF16_MUL_COLS
        + bf16_unary_rows * BF16_UNARY_COLS
        + bf16_div_rows * BF16_DIV_COLS
        + bf16_add_sub_rows * BF16_ADD_SUB_COLS
        + global_memory_boundary_rows * GLOBAL_MEMORY_BOUNDARY_COLS
        + public_values_rows * NUM_PUBLIC_VALUES_COLS;
    let stacking_area = 1 << LOG_STACKING_HEIGHT;
    let preprocessed_area = preprocessed_area.next_multiple_of(stacking_area);
    let main_area = main_area.next_multiple_of(stacking_area);

    VerillmTraceEstimate {
        memory_const_rows,
        memory_var_rows,
        base_alu_rows,
        poseidon2_rows,
        bf16_lookup_rows,
        bf16_mul_rows,
        bf16_unary_rows,
        bf16_div_rows,
        bf16_add_sub_rows,
        global_memory_boundary_rows,
        public_values_rows,
        preprocessed_area,
        main_area,
        total_area: preprocessed_area + main_area,
    }
}

const fn balanced_range(total: usize, index: usize, shards: usize) -> EventRange {
    EventRange { start: total * index / shards, end: total * (index + 1) / shards }
}

fn balanced_ranges(
    counts: RecursionAirEventCount,
    index: usize,
    shards: usize,
) -> RecursionEventRanges {
    RecursionEventRanges {
        mem_const: balanced_range(counts.mem_const_events, index, shards),
        mem_var: balanced_range(counts.mem_var_events, index, shards),
        base_alu: balanced_range(counts.base_alu_events, index, shards),
        poseidon2_wide: balanced_range(counts.poseidon2_wide_events, index, shards),
        bf16_mul: balanced_range(counts.bf16_mul_events, index, shards),
        bf16_unary: balanced_range(counts.bf16_unary_events, index, shards),
        bf16_div: balanced_range(counts.bf16_div_events, index, shards),
        bf16_add_sub: balanced_range(counts.bf16_add_sub_events, index, shards),
        commit_pv_hash: balanced_range(counts.commit_pv_hash_events, index, shards),
    }
}

/// Find the smallest balanced shard count satisfying both configured limits.
///
/// Every chip vector is split independently into contiguous ranges. This is intentionally
/// independent of LayerNorm/attention/MLP boundaries: the resulting leaf is a slice of the global
/// trace, not a standalone semantic circuit.
#[must_use]
pub fn plan_event_shards(counts: RecursionAirEventCount, limits: ShardLimits) -> EventShardPlan {
    // Validate the machine selection even if the first estimate already fits.
    let _ = estimate_verillm_trace(counts);
    let largest_event_vector = [
        counts.mem_const_events,
        counts.mem_var_events,
        counts.base_alu_events,
        counts.poseidon2_wide_events,
        counts.bf16_mul_events,
        counts.bf16_unary_events,
        counts.bf16_div_events,
        counts.bf16_add_sub_events,
        counts.commit_pv_hash_events,
    ]
    .into_iter()
    .max()
    .unwrap_or_default()
    .max(1);

    for shard_count in 1..=largest_event_vector {
        let shards = (0..shard_count)
            .map(|index| {
                let ranges = balanced_ranges(counts, index, shard_count);
                let estimate = estimate_verillm_trace(ranges.event_counts());
                EventShard { index, ranges, estimate }
            })
            .collect::<Vec<_>>();
        if shards.iter().all(|shard| shard.estimate.fits(limits)) {
            return EventShardPlan { global_counts: counts, limits, shards };
        }
    }
    panic!("one scalar event does not fit in the configured shard limits");
}

/// The estimated BF16 portion of one leaf trace.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Bf16TraceEstimate {
    pub mul_rows: usize,
    pub add_sub_rows: usize,
    pub unary_rows: usize,
    pub div_rows: usize,
    pub arithmetic_trace_area: usize,
    pub estimated_trace_area: usize,
}

impl Bf16TraceEstimate {
    #[must_use]
    pub const fn max_rows(self) -> usize {
        let lhs = if self.mul_rows > self.add_sub_rows { self.mul_rows } else { self.add_sub_rows };
        let rhs = if self.unary_rows > self.div_rows { self.unary_rows } else { self.div_rows };
        if lhs > rhs {
            lhs
        } else {
            rhs
        }
    }
}

/// A uniform plan. Equal shard widths let every leaf reuse one program and one proving key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NaturalShardPlan {
    pub total_units: usize,
    pub units_per_shard: usize,
    pub shard_count: usize,
    pub estimate: Bf16TraceEstimate,
    pub limits: ShardLimits,
}

const fn padded_rows(events: usize, lanes: usize) -> usize {
    if events == 0 {
        0
    } else {
        events.div_ceil(lanes).next_multiple_of(32)
    }
}

/// Estimate the BF16 trace generated by `units` complete semantic units.
#[must_use]
pub fn estimate_bf16_trace(
    events: Bf16EventsPerUnit,
    units: usize,
    limits: ShardLimits,
) -> Bf16TraceEstimate {
    assert!(units > 0, "a leaf shard must contain at least one semantic unit");
    let mul_rows = padded_rows(events.mul.checked_mul(units).unwrap(), NUM_BF16_MUL_EVENTS_PER_ROW);
    let add_sub_rows =
        padded_rows(events.add_sub.checked_mul(units).unwrap(), NUM_BF16_ADD_SUB_EVENTS_PER_ROW);
    let unary_rows = padded_rows(events.unary.checked_mul(units).unwrap(), 1);
    let div_rows = padded_rows(events.div.checked_mul(units).unwrap(), 1);
    let arithmetic_trace_area = mul_rows * (BF16_MUL_COLS + BF16_MUL_PREPROCESSED_COLS)
        + add_sub_rows * (BF16_ADD_SUB_COLS + BF16_ADD_SUB_PREPROCESSED_COLS)
        + unary_rows * (BF16_UNARY_COLS + BF16_UNARY_PREPROCESSED_COLS)
        + div_rows * (BF16_DIV_COLS + BF16_DIV_PREPROCESSED_COLS);
    Bf16TraceEstimate {
        mul_rows,
        add_sub_rows,
        unary_rows,
        div_rows,
        arithmetic_trace_area,
        estimated_trace_area: arithmetic_trace_area + limits.non_arithmetic_area_reserve,
    }
}

/// Pack equal contiguous semantic units into one leaf.
///
/// The selected width must divide `total_units`; this deliberately trades a small amount of
/// packing efficiency for one circuit shape and one reusable proving key across all leaves.
#[must_use]
pub fn plan_uniform_natural_shards(
    total_units: usize,
    events: Bf16EventsPerUnit,
    limits: ShardLimits,
) -> NaturalShardPlan {
    assert!(total_units > 0, "a stage must contain at least one semantic unit");
    let mut largest_fitting = 0;
    for units in 1..=total_units {
        let estimate = estimate_bf16_trace(events, units, limits);
        if estimate.max_rows() <= limits.max_rows()
            && estimate.estimated_trace_area <= limits.max_trace_area
        {
            largest_fitting = units;
        }
    }
    assert!(largest_fitting > 0, "one semantic unit does not fit in the configured leaf limits");

    let units_per_shard = (1..=largest_fitting)
        .rev()
        .find(|units| total_units.is_multiple_of(*units))
        .expect("one always divides the number of semantic units");
    let estimate = estimate_bf16_trace(events, units_per_shard, limits);
    NaturalShardPlan {
        total_units,
        units_per_shard,
        shard_count: total_units / units_per_shard,
        estimate,
        limits,
    }
}

/// Plan a BF16 linear layer using a complete output column as the indivisible unit.
#[must_use]
pub fn plan_linear_output_columns(
    sequence_length: usize,
    input_width: usize,
    output_width: usize,
    unary_events_per_output: usize,
    limits: ShardLimits,
) -> NaturalShardPlan {
    assert!(sequence_length > 0);
    assert!(input_width > 0);
    assert!(output_width > 0);
    let events = Bf16EventsPerUnit {
        mul: sequence_length * input_width,
        add_sub: sequence_length * (input_width - 1),
        unary: sequence_length * unary_events_per_output,
        div: 0,
    };
    plan_uniform_natural_shards(output_width, events, limits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_linear_stages_choose_natural_output_column_widths() {
        let limits = ShardLimits::full();
        let c_proj = plan_linear_output_columns(30, 768, 768, 0, limits);
        assert_eq!((c_proj.units_per_shard, c_proj.shard_count), (256, 3));

        let expansion = plan_linear_output_columns(30, 768, 2304, 1, limits);
        assert_eq!((expansion.units_per_shard, expansion.shard_count), (256, 9));

        let projection = plan_linear_output_columns(30, 2304, 768, 0, limits);
        assert_eq!((projection.units_per_shard, projection.shard_count), (64, 12));
    }

    #[test]
    fn attention_head_is_already_a_full_natural_unit() {
        let events =
            Bf16EventsPerUnit { mul: 4_529_745, add_sub: 4_568_085, unary: 23_535, div: 525 };
        let plan = plan_uniform_natural_shards(12, events, ShardLimits::full());
        assert_eq!((plan.units_per_shard, plan.shard_count), (1, 12));
    }

    #[test]
    fn area_limit_can_cut_before_the_height_limit() {
        let limits = ShardLimits {
            max_log_rows: 19,
            max_trace_area: 100_000,
            non_arithmetic_area_reserve: 0,
        };
        let events = Bf16EventsPerUnit { mul: 1_200, ..Bf16EventsPerUnit::default() };
        let plan = plan_uniform_natural_shards(16, events, limits);
        assert!(plan.estimate.estimated_trace_area <= limits.max_trace_area);
        assert!(plan.units_per_shard < 16);
    }

    #[test]
    fn global_event_plan_covers_every_event_once() {
        let counts = RecursionAirEventCount {
            mem_const_events: 23_040,
            mem_var_events: 5_901_672,
            base_alu_events: 16,
            poseidon2_wide_events: 743_488,
            bf16_mul_events: 177_764_880,
            bf16_unary_events: 120_840,
            bf16_div_events: 360,
            bf16_add_sub_events: 177_643_560,
            ..RecursionAirEventCount::default()
        };
        let plan = plan_event_shards(counts, ShardLimits::core_style());
        assert!(plan.shards.len() > 1);
        assert!(plan.shards.iter().all(|shard| shard.estimate.fits(plan.limits)));

        let assert_coverage = |ranges: Vec<EventRange>, total: usize| {
            assert_eq!(ranges.first().unwrap().start, 0);
            assert_eq!(ranges.last().unwrap().end, total);
            for pair in ranges.windows(2) {
                assert_eq!(pair[0].end, pair[1].start);
            }
        };
        assert_coverage(
            plan.shards.iter().map(|shard| shard.ranges.mem_var).collect(),
            counts.mem_var_events,
        );
        assert_coverage(
            plan.shards.iter().map(|shard| shard.ranges.bf16_mul).collect(),
            counts.bf16_mul_events,
        );
        assert_coverage(
            plan.shards.iter().map(|shard| shard.ranges.bf16_add_sub).collect(),
            counts.bf16_add_sub_events,
        );
    }

    #[test]
    fn event_plan_uses_area_even_when_each_chip_height_fits() {
        let counts = RecursionAirEventCount {
            // Each vector fits below 2^22 rows with 12 lanes, but their combined area does not fit
            // into one Core-style shard.
            bf16_mul_events: 40_000_000,
            bf16_add_sub_events: 40_000_000,
            ..RecursionAirEventCount::default()
        };
        let one = estimate_verillm_trace(counts);
        assert!(one.max_rows() <= ShardLimits::core_style().max_rows());
        assert!(one.total_area > ShardLimits::core_style().max_trace_area);
        assert!(plan_event_shards(counts, ShardLimits::core_style()).shards.len() > 1);
    }
}
