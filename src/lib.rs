#![warn(clippy::std_instead_of_core, clippy::std_instead_of_alloc)]
#![no_std]
use core::slice::Iter;

mod float;
pub use float::Float;

/// Simple fixed size ringbuffer with fast averaging.
pub struct SmoothBuffer<const CAP: usize, T: Float> {
    data: [T; CAP],
    head: usize,
    sum: Option<T>,
    max: Option<T>,
    min: Option<T>,
    filled_len: usize,
}

impl<const CAP: usize, T: Float> Default for SmoothBuffer<CAP, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAP: usize, T: Float> SmoothBuffer<CAP, T> {
    /// Creates a new, empty buffer.
    pub fn new() -> Self {
        SmoothBuffer {
            data: [T::default(); CAP],
            head: 0,
            sum: None,
            max: None,
            min: None,
            filled_len: 0,
        }
    }

    /// Creates a new buffer pre-populated with a value, filled to capacity.
    pub fn pre_filled(value: T) -> Self {
        SmoothBuffer {
            data: [value; CAP],
            head: CAP - 1,
            sum: Some(value * T::from_usize(CAP)),
            max: Some(value),
            min: Some(value),
            filled_len: CAP,
        }
    }

    /// Fast! Sum is always kept up to date on push. No need to iterate.
    pub fn average(&self) -> T {
        if self.filled_len > 0 {
            return self.sum.unwrap_or(T::zero()) / T::from_usize(self.filled_len);
        }
        T::zero()
    }

    /// Resets buffer to its default empty state.
    pub fn clear(&mut self) {
        for n in 0..self.data.len() {
            self.data[n] = T::zero();
        }
        self.sum = None;
        self.max = None;
        self.min = None;
        self.filled_len = 0;
        self.head = 0;
    }

    /// True is buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.filled_len == 0
    }

    /// The largest value so far, if any.
    pub fn max(&self) -> T {
        self.max.unwrap_or(T::zero())
    }

    /// The smallest value so far, if any.
    pub fn min(&self) -> T {
        self.min.unwrap_or(T::zero())
    }

    /// The maximum number of items. Older items are discarded in favor of newer ones
    /// if capacity is exceeded.
    pub fn capacity(&self) -> usize {
        CAP
    }

    /// Current value count, will always be lower or equal to capacity.
    pub fn len(&self) -> usize {
        self.filled_len
    }

    /// Push a single value.
    pub fn push(&mut self, value: T) {
        match self.max {
            None => self.max = Some(value),
            Some(max) => self.max = Some(T::get_max(max, value)),
        }
        match self.min {
            None => self.min = Some(value),
            Some(min) => self.min = Some(T::get_min(min, value)),
        }
        match self.sum {
            None => self.sum = Some(value),
            Some(sum) => self.sum = Some(sum - self.data[self.head] + value),
        }

        // Push data into storage
        self.data[self.head] = value;
        self.head += 1;
        if self.head == CAP {
            self.head = 0
        }
        if self.filled_len < CAP {
            self.filled_len += 1;
        }
    }

    /// Pushes multiple values at once.
    pub fn push_slice(&mut self, slice: &[T]) {
        for item in slice {
            self.push(*item);
        }
    }

    /// Iterates through all values. Order of retrieval will likely NOT match order of input.
    pub fn iter(&self) -> Iter<T> {
        self.data[0..self.filled_len].iter()
    }

    /// Gaussian smoothing. Much slower than a simple average, will actually
    /// iterate through all values and return a weighted sum.
    pub fn gaussian_filter(&self) -> T {
        if CAP == 0 {
            return T::zero();
        }

        // Standard deviation for the Gaussian kernel
        let sigma = T::from_usize(CAP) / T::four();
        let mut weights = [T::zero(); CAP];

        // Calculate Gaussian weights
        let mut total_weight = T::zero();
        let center = T::from_usize(CAP - 1) / T::two();
        for i in 0..CAP {
            let distance = T::from_usize(i) - center;
            let weight = T::exp(-distance * distance / (T::two() * sigma * sigma));
            weights[i] = weight;
            total_weight += weight;
        }

        // Normalize weights
        for weight in weights.iter_mut() {
            *weight /= total_weight;
        }

        // Compute the weighted sum
        let mut sum = T::zero();
        self.data.iter()
            .zip(weights.iter())
            .for_each(|(value, weight)| sum += *value * *weight);
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const MARGIN: f64 = 0.000001;

    #[test]
    fn create_and_push() {
        const CAP: usize = 10;
        let mut buf = SmoothBuffer::<CAP, f32>::new();
        for _ in 0..5 {
            buf.push(10.0);
        }

        assert_eq!(buf.capacity(), CAP);
        assert_eq!(buf.len(), 5);
        assert_eq!(buf.average(), 10.0);

        for _ in 0..10 {
            buf.push(5.0);
        }
        assert_eq!(buf.len(), CAP);
        assert_eq!(buf.average(), 5.0);
    }

    #[test]
    fn clearing() {
        let mut buf = SmoothBuffer::<10, f32>::new();
        for n in 0..buf.capacity() {
            buf.push(n as f32);
        }
        buf.clear();
        assert_eq!(buf.capacity(), 10);
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.average(), 0.0);
        assert_eq!(buf.iter().next(), None);
    }

    #[test]
    fn iteration() {
        let mut buf = SmoothBuffer::<10, f64>::new();
        let len = 7;
        for n in 0..len {
            buf.push(n as f64);
        }

        for (i, value) in buf.iter().enumerate() {
            assert_eq!(i as f64, *value);
        }

        assert!(buf.iter().len() == len);
    }

    #[test]
    fn gaussian_smoothing_simple() {
        let mut buf = SmoothBuffer::<10, f64>::new();
        for _ in 0..100 {
            buf.push(3.0);
        }
        // Smoothed value won't be exactly the same! Will be correct to a few decimal places though
        // println!("{}", buf.gaussian_filter(0.5));
        assert!(buf.gaussian_filter() - 3.0 < MARGIN);
    }

    #[test]
    fn gaussian_smoothing_with_negative_values() {
        let mut buf = SmoothBuffer::<10, f64>::new();
        let mid = buf.len() / 2;
        for v in 0..buf.len() {
            buf.push(if v < mid {
                1.0
            } else if v > mid {
                -1.0
            } else {
                0.0
            });
        }
        // println!("{}", buf.gaussian_filter());
        assert!(buf.gaussian_filter().abs() < MARGIN);
    }

    #[test]
    fn gaussian_smoothing_slice() {
        let mut buf = SmoothBuffer::<10, f64>::new();
        buf.push_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]);
        assert!(buf.gaussian_filter() - 0.55 < MARGIN);
    }

    #[test]
    fn pre_filled_buffer() {
        fn test_value(x:f64){
            // println!("testing {}", x);
            let buf = SmoothBuffer::<10, f64>::pre_filled(x);
            assert!(buf.len() == 10);
            assert!(buf.gaussian_filter() - x < MARGIN);
            assert!(buf.average() - x < MARGIN);
        }

        for n in 0 ..= 10 {
            test_value(n as f64 / 10.0);
        }
    }

    #[test]
    fn progressive_fill() {
        let mut buf = SmoothBuffer::<10, f64>::pre_filled(0.0);
        // println!("{}", buf.gaussian_filter());
        assert!(buf.gaussian_filter() < MARGIN);
        for _n in 0..10 {
            buf.push(1.0);
            // println!("{:.2}", buf.gaussian_filter());
        }
        assert!(buf.gaussian_filter() - 1.0 < MARGIN);
    }
}
