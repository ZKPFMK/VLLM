use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use slop_algebra::{ExtensionField, Field};
use slop_alloc::{Backend, CpuBackend};
use slop_challenger::FieldChallenger;
use slop_multilinear::{Mle, MleEval, Point};
use slop_sumcheck::PartialSumcheckProof;

/// Lookup challenges shared by every trace shard in one batched execution.
///
/// A batch prover must derive these challenges only after committing to every shard trace.  This
/// lets individual shard proofs carry a non-zero local LogUp residual while a batch verifier checks
/// that the residuals cancel globally.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct LogUpGkrChallenges<EF> {
    /// The random offset used in every lookup denominator.
    pub alpha: EF,
    /// Randomness used to compress an interaction tuple.
    pub beta_seed: Point<EF>,
    /// Randomness used by interactions declared in public values.
    pub public_values_challenge: EF,
}

impl<EF> LogUpGkrChallenges<EF> {
    /// Sample a common challenge set from an already initialized batch transcript.
    ///
    /// The transcript must contain every preprocessed and main commitment, the ordered shard
    /// shapes, and a domain separator before this method is called.
    pub fn sample<F, Challenger>(challenger: &mut Challenger, beta_seed_dimension: usize) -> Self
    where
        F: Field,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    {
        Self {
            alpha: challenger.sample_ext_element::<EF>(),
            beta_seed: (0..beta_seed_dimension)
                .map(|_| challenger.sample_ext_element::<EF>())
                .collect(),
            public_values_challenge: challenger.sample_ext_element::<EF>(),
        }
    }
}

/// The output of the log-up GKR circuit.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound(serialize = "Mle<EF, B>: Serialize", deserialize = "Mle<EF, B>: Deserialize<'de>"))]
pub struct LogUpGkrOutput<EF, B: Backend = CpuBackend> {
    /// Numerator
    pub numerator: Mle<EF, B>,
    /// Denominator
    pub denominator: Mle<EF, B>,
}

/// The proof for a single round of the log-up GKR circuit.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LogupGkrRoundProof<EF> {
    /// The numerator of the numerator with last coordinate being 0.
    pub numerator_0: EF,
    /// The numerator of the numerator with last coordinate being 1.
    pub numerator_1: EF,
    /// The denominator of the denominator with last coordinate being 0.
    pub denominator_0: EF,
    /// The denominator of the denominator with last coordinate being 1.
    pub denominator_1: EF,
    /// The sumcheck proof for the round.
    pub sumcheck_proof: PartialSumcheckProof<EF>,
}

/// The proof for the log-up GKR circuit.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LogupGkrProof<F, EF> {
    /// The output of the circuit.
    pub circuit_output: LogUpGkrOutput<EF>,
    /// The proof for each round.
    pub round_proofs: Vec<LogupGkrRoundProof<EF>>,
    /// The evaluations for each chip.
    pub logup_evaluations: LogUpEvaluations<EF>,
    /// The grinding witness.
    pub witness: F,
}

/// The evaluations for a chip
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChipEvaluation<EF> {
    /// The evaluations of the main trace.
    pub main_trace_evaluations: MleEval<EF>,
    /// The evaluations of the preprocessed trace.
    pub preprocessed_trace_evaluations: Option<MleEval<EF>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// The data passed from the GKR prover to the zerocheck prover.
pub struct LogUpEvaluations<EF> {
    /// The point at which the evaluations are made.
    pub point: Point<EF>,
    /// The evaluations for each chip.
    pub chip_openings: BTreeMap<String, ChipEvaluation<EF>>,
}
