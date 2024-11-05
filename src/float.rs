use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use libm::{exp, expf};

/// A floating point type. Currently implemented for f32 and f64.
pub trait Float:
    Default
    + PartialEq
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
    fn exp(a: Self) -> Self;
}

/// Takes in the type and the necessary exponential function for that type.
macro_rules! impl_float {
    ($t:ty, $exp_func:ident) => {
        impl Float for $t {
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

            #[inline(always)]
            fn exp(a: Self) -> Self {
                $exp_func(a)
            }
        }
    };
}

impl_float!(f32, expf);
impl_float!(f64, exp);
