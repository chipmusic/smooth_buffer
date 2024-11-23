use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A signed Number type. Currently implemented for f32, f64, i8, i16, i32 and i64.
pub trait Num:
    Default
    + PartialEq
    + PartialOrd
    + Copy
    + AddAssign
    + MulAssign
    + SubAssign
    + DivAssign
    + Add<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn zero() -> Self;
    fn two() -> Self;
    fn four() -> Self;
    fn max() -> Self;
    fn from_usize(value: usize) -> Self;
    fn get_max(a: Self, b: Self) -> Self;
    fn get_min(a: Self, b: Self) -> Self;
}

/// Takes in the type and the necessary exponential function for that type.
macro_rules! impl_num {
    ($t:ty) => {
        impl Num for $t {
            #[inline(always)]
            fn zero() -> Self {
                0
            }

            #[inline(always)]
            fn two() -> Self {
                2
            }

            #[inline(always)]
            fn four() -> Self {
                4
            }

            #[inline(always)]
            fn max() -> Self {
                Self::MAX
            }

            #[inline(always)]
            fn from_usize(value: usize) -> Self {
                value as Self
            }

            #[inline(always)]
            fn get_max(a: Self, b: Self) -> Self {
                a.max(b)
            }

            #[inline(always)]
            fn get_min(a: Self, b: Self) -> Self {
                a.min(b)
            }
        }
    };
}

impl_num!(i8);
impl_num!(i16);
impl_num!(i32);
impl_num!(i64);

pub trait Float: Num {
    fn floor(a: Self) -> Self;
    fn ceil(a: Self) -> Self;
    fn round(a: Self) -> Self;
    fn exp(a: Self) -> Self;
}

/// Takes in the type and the necessary exponential function for that type.
macro_rules! impl_float {
    ($t:ty, $exp_fn:ident, $floor_fn:ident, $ceil_fn:ident, $round_fn:ident) => {
        impl Num for $t {
            #[inline(always)]
            fn zero() -> Self {
                0.0
            }

            #[inline(always)]
            fn two() -> Self {
                2.0
            }

            #[inline(always)]
            fn four() -> Self {
                4.0
            }

            #[inline(always)]
            fn max() -> Self {
                Self::MAX
            }

            #[inline(always)]
            fn from_usize(value: usize) -> Self {
                value as Self
            }

            #[inline(always)]
            fn get_max(a: Self, b: Self) -> Self {
                a.max(b)
            }

            #[inline(always)]
            fn get_min(a: Self, b: Self) -> Self {
                a.min(b)
            }
        }

        impl Float for $t {
            #[inline(always)]
            fn floor(a: Self) -> Self {
                $floor_fn(a)
            }

            #[inline(always)]
            fn ceil(a: Self) -> Self {
                $ceil_fn(a)
            }

            #[inline(always)]
            fn round(a: Self) -> Self {
                $round_fn(a)
            }


            #[inline(always)]
            fn exp(a: Self) -> Self {
                $exp_fn(a)
            }
        }
    };
}

use libm::{exp, expf, floor, floorf, ceil, ceilf, round, roundf};

impl_float!(f32, expf, floorf, ceilf, roundf);
impl_float!(f64, exp, floor, ceil, round);
