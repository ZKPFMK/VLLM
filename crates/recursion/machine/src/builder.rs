use std::iter::once;

use slop_air::AirBuilderWithPublicValues;
use slop_algebra::AbstractField;
use sp1_hypercube::{
    air::{AirInteraction, BaseAirBuilder, InteractionScope, MachineAirBuilder},
    InteractionKind,
};
use sp1_recursion_executor::{
    Address, Block, BF16_LOOKUP_INIT, BF16_LOOKUP_MUL, BF16_LOOKUP_SHARED, BF16_LSHIFT_PREFIX,
    BF16_MANTISSA_BITS, BF16_ROUND_PREFIX, BF16_RSHIFT_PREFIX,
};

/// A trait which contains all helper methods for building SP1 recursion machine AIRs.
pub trait SP1RecursionAirBuilder: MachineAirBuilder + RecursionAirBuilder {}

impl<AB: AirBuilderWithPublicValues + RecursionAirBuilder> SP1RecursionAirBuilder for AB {}
impl<AB: BaseAirBuilder> RecursionAirBuilder for AB {}

pub trait RecursionAirBuilder: BaseAirBuilder {
    /// Low-level BF16 lookup interaction. Prefer one of the operation-specific helpers below.
    fn send_bf16_lookup<Op, Input, Out0, Out1, Out2>(
        &mut self,
        opcode: Op,
        input: Input,
        out0: Out0,
        out1: Out1,
        out2: Out2,
        mult: impl Into<Self::Expr>,
    ) where
        Op: Into<Self::Expr>,
        Input: Into<Self::Expr>,
        Out0: Into<Self::Expr>,
        Out1: Into<Self::Expr>,
        Out2: Into<Self::Expr>,
    {
        self.send(
            AirInteraction::new(
                vec![opcode.into(), input.into(), out0.into(), out1.into(), out2.into()],
                mult.into(),
                InteractionKind::Bf16,
            ),
            InteractionScope::Local,
        );
    }

    /// Look up the circuit representation `(sign, exponent, mantissa)` of a raw BF16 value.
    fn send_bf16_init<Raw, Sign, Exponent, Mantissa>(
        &mut self,
        raw: Raw,
        sign: Sign,
        exponent: Exponent,
        mantissa: Mantissa,
        mult: impl Into<Self::Expr>,
    ) where
        Raw: Into<Self::Expr>,
        Sign: Into<Self::Expr>,
        Exponent: Into<Self::Expr>,
        Mantissa: Into<Self::Expr>,
    {
        self.send_bf16_lookup(
            Self::Expr::from_canonical_u8(BF16_LOOKUP_INIT),
            raw,
            sign,
            exponent,
            mantissa,
            mult,
        );
    }

    /// Look up `Mul(lhs_mantissa || rhs_mantissa)`.
    fn send_bf16_mul<Lhs, Rhs, Product>(
        &mut self,
        lhs_mantissa: Lhs,
        rhs_mantissa: Rhs,
        product: Product,
        mult: impl Into<Self::Expr>,
    ) where
        Lhs: Into<Self::Expr>,
        Rhs: Into<Self::Expr>,
        Product: Into<Self::Expr>,
    {
        let input = lhs_mantissa.into()
            * Self::Expr::from_canonical_u16(1 << (BF16_MANTISSA_BITS + 1))
            + rhs_mantissa.into();
        self.send_bf16_lookup(
            Self::Expr::from_canonical_u8(BF16_LOOKUP_MUL),
            input,
            product,
            Self::Expr::zero(),
            Self::Expr::zero(),
            mult,
        );
    }

    /// Look up `Exp(exponent)`.
    fn send_bf16_exp<Exponent, Output>(
        &mut self,
        exponent: Exponent,
        output: Output,
        mult: impl Into<Self::Expr>,
    ) where
        Exponent: Into<Self::Expr>,
        Output: Into<Self::Expr>,
    {
        let input = exponent.into() + Self::Expr::from_canonical_u16(1 << 12);
        self.send_bf16_lookup(
            Self::Expr::from_canonical_u8(BF16_LOOKUP_SHARED),
            input,
            output,
            Self::Expr::zero(),
            Self::Expr::zero(),
            mult,
        );
    }

    /// Look up `Clamp(exponent)`.
    fn send_bf16_clamp<Exponent, Output>(
        &mut self,
        exponent: Exponent,
        output: Output,
        mult: impl Into<Self::Expr>,
    ) where
        Exponent: Into<Self::Expr>,
        Output: Into<Self::Expr>,
    {
        let input = exponent.into() + Self::Expr::from_canonical_u16(3 << 12);
        self.send_bf16_lookup(
            Self::Expr::from_canonical_u8(BF16_LOOKUP_SHARED),
            input,
            output,
            Self::Expr::zero(),
            Self::Expr::zero(),
            mult,
        );
    }

    /// Look up `Round(clamp, mantissa)`.
    fn send_bf16_round<Clamp, Mantissa, Output>(
        &mut self,
        clamp: Clamp,
        mantissa: Mantissa,
        output: Output,
        mult: impl Into<Self::Expr>,
    ) where
        Clamp: Into<Self::Expr>,
        Mantissa: Into<Self::Expr>,
        Output: Into<Self::Expr>,
    {
        let input = clamp.into() * Self::Expr::from_canonical_u16(1 << (BF16_MANTISSA_BITS + 1))
            + mantissa.into()
            + Self::Expr::from_canonical_u16(BF16_ROUND_PREFIX);
        self.send_bf16_lookup(
            Self::Expr::from_canonical_u8(BF16_LOOKUP_SHARED),
            input,
            output,
            Self::Expr::zero(),
            Self::Expr::zero(),
            mult,
        );
    }

    /// Look up `LShift(shift, sign, mantissa)`.
    fn send_bf16_lshift<Shift, Sign, Mantissa, Output>(
        &mut self,
        shift: Shift,
        sign: Sign,
        mantissa: Mantissa,
        output: Output,
        mult: impl Into<Self::Expr>,
    ) where
        Shift: Into<Self::Expr>,
        Sign: Into<Self::Expr>,
        Mantissa: Into<Self::Expr>,
        Output: Into<Self::Expr>,
    {
        let input = shift.into() * Self::Expr::from_canonical_u16(1 << 10)
            + sign.into() * Self::Expr::from_canonical_u16(1 << 9)
            + mantissa.into()
            + Self::Expr::from_canonical_u16(BF16_LSHIFT_PREFIX);
        self.send_bf16_lookup(
            Self::Expr::from_canonical_u8(BF16_LOOKUP_SHARED),
            input,
            output,
            Self::Expr::zero(),
            Self::Expr::zero(),
            mult,
        );
    }

    /// Look up `RShift(offset, value)`, where `offset` selects a shift of `M + offset` bits.
    fn send_bf16_rshift<Value, Output>(
        &mut self,
        offset: u16,
        value: Value,
        output: Output,
        mult: impl Into<Self::Expr>,
    ) where
        Value: Into<Self::Expr>,
        Output: Into<Self::Expr>,
    {
        assert!(offset < 4, "BF16 RShift offset must fit in two bits");
        let input =
            value.into() + Self::Expr::from_canonical_u16(BF16_RSHIFT_PREFIX | (offset << 12));
        self.send_bf16_lookup(
            Self::Expr::from_canonical_u8(BF16_LOOKUP_SHARED),
            input,
            output,
            Self::Expr::zero(),
            Self::Expr::zero(),
            mult,
        );
    }

    /// Low-level receiver used by the preprocessed BF16 lookup table.
    fn receive_bf16_lookup<Op, Input, Out0, Out1, Out2>(
        &mut self,
        opcode: Op,
        input: Input,
        out0: Out0,
        out1: Out1,
        out2: Out2,
        mult: impl Into<Self::Expr>,
    ) where
        Op: Into<Self::Expr>,
        Input: Into<Self::Expr>,
        Out0: Into<Self::Expr>,
        Out1: Into<Self::Expr>,
        Out2: Into<Self::Expr>,
    {
        self.receive(
            AirInteraction::new(
                vec![opcode.into(), input.into(), out0.into(), out1.into(), out2.into()],
                mult.into(),
                InteractionKind::Bf16,
            ),
            InteractionScope::Local,
        );
    }

    fn send_single<E: Into<Self::Expr>>(
        &mut self,
        addr: Address<E>,
        val: E,
        mult: impl Into<Self::Expr>,
    ) {
        let mut padded_value = core::array::from_fn(|_| Self::Expr::zero());
        padded_value[0] = val.into();
        self.send_block(Address(addr.0.into()), Block(padded_value), mult)
    }

    fn send_block<E: Into<Self::Expr>>(
        &mut self,
        addr: Address<E>,
        val: Block<E>,
        mult: impl Into<Self::Expr>,
    ) {
        self.send(
            AirInteraction::new(
                once(addr.0).chain(val).map(Into::into).collect(),
                mult.into(),
                InteractionKind::Memory,
            ),
            InteractionScope::Local,
        );
    }

    fn receive_single<E: Into<Self::Expr>>(
        &mut self,
        addr: Address<E>,
        val: E,
        mult: impl Into<Self::Expr>,
    ) {
        let mut padded_value = core::array::from_fn(|_| Self::Expr::zero());
        padded_value[0] = val.into();
        self.receive_block(Address(addr.0.into()), Block(padded_value), mult)
    }

    fn receive_block<E: Into<Self::Expr>>(
        &mut self,
        addr: Address<E>,
        val: Block<E>,
        mult: impl Into<Self::Expr>,
    ) {
        self.receive(
            AirInteraction::new(
                once(addr.0).chain(val).map(Into::into).collect(),
                mult.into(),
                InteractionKind::Memory,
            ),
            InteractionScope::Local,
        );
    }
}
