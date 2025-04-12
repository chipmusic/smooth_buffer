#![warn(clippy::std_instead_of_core, clippy::std_instead_of_alloc)]
#![no_std]

mod num;
pub use num::{Float, Num};

/// Simple fixed size ringbuffer with fast averaging.
pub struct SmoothBuffer<const CAP: usize, T: Num> {
    data: [T; CAP],
    head: usize,
    sum: Option<T>,
    max: Option<T>,
    min: Option<T>,
    filled_len: usize,
    dirty_minmax: bool,
}

impl<const CAP: usize, T: Num> Default for SmoothBuffer<CAP, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAP: usize, T: Num> SmoothBuffer<CAP, T> {
    /// Creates a new, empty buffer.
    pub fn new() -> Self {
        assert!(
            CAP > 0,
            "SmoothBuffer Error: Capacity must be larger than zero"
        );
        SmoothBuffer {
            data: [T::default(); CAP],
            head: 0,
            sum: None,
            max: None,
            min: None,
            filled_len: 0,
            dirty_minmax: false,
        }
    }

    /// Creates a new buffer pre-populated with a value, filled to capacity.
    pub fn pre_filled(value: T) -> Self {
        assert!(
            CAP > 0,
            "SmoothBuffer Error: Capacity must be larger than zero"
        );
        SmoothBuffer {
            data: [value; CAP],
            head: CAP - 1,
            sum: Some(value * T::from_usize(CAP)),
            max: Some(value),
            min: Some(value),
            filled_len: CAP,
            dirty_minmax: false,
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
    pub fn max(&mut self) -> T {
        if self.dirty_minmax {
            // Recalculate max only when needed
            self.recalculate_minmax();
        }
        self.max.unwrap_or(T::zero())
    }

    /// The smallest value so far, if any.
    pub fn min(&mut self) -> T {
        if self.dirty_minmax {
            // Recalculate min only when needed
            self.recalculate_minmax();
        }
        self.min.unwrap_or(T::zero())
    }

    fn recalculate_minmax(&mut self) {
        if self.filled_len == 0 {
            self.min = None;
            self.max = None;
            return;
        }

        // First find the index of the oldest element
        let start_idx = if self.filled_len < CAP {
            0
        } else {
            (self.head + CAP - self.filled_len) % CAP
        };

        // Initialize with the first valid element
        let mut new_min = self.data[start_idx];
        let mut new_max = self.data[start_idx];

        // Check all valid elements
        for i in 1..self.filled_len {
            let idx = (start_idx + i) % CAP;
            new_min = T::get_min(new_min, self.data[idx]);
            new_max = T::get_max(new_max, self.data[idx]);
        }

        self.min = Some(new_min);
        self.max = Some(new_max);
        self.dirty_minmax = false;
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
        // Fast path for sum calculation
        if self.filled_len < CAP {
            self.sum = match self.sum {
                None => Some(value),
                Some(sum) => Some(sum + value),
            };
        } else {
            // Buffer is full, subtract old value and add new
            self.sum = Some(self.sum.unwrap_or(T::zero()) - self.data[self.head] + value);
        }

        // Push data into storage
        self.data[self.head] = value;
        self.head = (self.head + 1) % CAP; // More efficient than if check
        if self.filled_len < CAP {
            self.filled_len += 1;
        }

        // Update min/max lazily
        if self.filled_len == CAP {
            // Only mark dirty if we're overwriting, not when adding
            self.dirty_minmax = true;
        } else {
            match self.max {
                None => self.max = Some(value),
                Some(max) if value > max => self.max = Some(value),
                _ => {}
            }

            match self.min {
                None => self.min = Some(value),
                Some(min) if value < min => self.min = Some(value),
                _ => {}
            }
        }
    }

    /// Pushes multiple values at once.
    pub fn push_slice(&mut self, slice: &[T]) {
        for item in slice {
            self.push(*item);
        }
    }

    /// Iterates through all values. Order of retrieval will likely NOT match order of input.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let head = self.head;
        let len = self.filled_len;
        let cap = CAP;

        (0..len).map(move |i| &self.data[(head + cap - len + i) % cap])
    }
}

impl<const CAP: usize, T: Float> SmoothBuffer<CAP, T> {
    /// Gaussian smoothing. Much slower than a simple average, will actually
    /// iterate through all values and return a weighted sum.
    pub fn gaussian_filter(&self) -> T {
        if self.filled_len == 0 {
            return T::zero();
        }

        // Standard deviation for the Gaussian kernel
        let sigma = T::from_usize(self.filled_len) / T::four();
        let center = T::from_usize(self.filled_len - 1) / T::two();

        // First pass: calculate all weights and their sum
        let mut weights = [T::zero(); CAP]; // Using fixed array instead of Vec
        let mut total_weight = T::zero();

        for i in 0..self.filled_len {
            let distance = T::from_usize(i) - center;
            let exp_term = -distance * distance / (T::two() * sigma * sigma);
            let weight = T::exp(exp_term);
            weights[i] = weight;
            total_weight += weight;
        }

        // Second pass: apply normalized weights to values
        let mut sum = T::zero();
        for i in 0..self.filled_len {
            // Calculate the actual index considering the circular buffer
            let idx = if self.filled_len < CAP {
                i
            } else {
                (self.head + CAP - self.filled_len + i) % CAP
            };

            // Use the pre-calculated weight
            let normalized_weight = weights[i] / total_weight;
            sum += self.data[idx] * normalized_weight;
        }

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
        fn test_value(x: f64) {
            // println!("testing {}", x);
            let buf = SmoothBuffer::<10, f64>::pre_filled(x);
            assert!(buf.len() == 10);
            assert!((buf.gaussian_filter() - x).abs() < MARGIN);
            assert!((buf.average() - x).abs() < MARGIN);
        }

        for n in 0..=10 {
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

    #[test]
    fn test_min_max_recalculation() {
        let mut buf = SmoothBuffer::<5, f64>::new();

        // Fill buffer with increasing values
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);
        buf.push(4.0);
        buf.push(5.0);

        // Initial min/max should be correct
        assert_eq!(buf.min(), 1.0);
        assert_eq!(buf.max(), 5.0);

        // Now overwrite the min value
        buf.push(2.5); // This overwrites 1.0

        // Min should be recalculated
        assert_eq!(buf.min(), 2.0);
        assert_eq!(buf.max(), 5.0);

        // Now overwrite the max value
        buf.push(3.5); // This overwrites 2.0
        buf.push(4.5); // This overwrites 3.0
        buf.push(3.0); // This overwrites 4.0
        buf.push(2.0); // This overwrites 5.0 (the max)

        // Max should be recalculated
        assert_eq!(buf.min(), 2.0);
        assert_eq!(buf.max(), 4.5);
    }

    #[test]
    fn test_buffer_wrapping() {
        let mut buf = SmoothBuffer::<3, i32>::new();

        // Fill buffer
        buf.push(1);
        buf.push(2);
        buf.push(3);

        // Check initial state
        assert_eq!(buf.average(), 2);

        // Push more values to wrap around
        buf.push(4); // Overwrites 1
        assert_eq!(buf.average(), 3);

        buf.push(5); // Overwrites 2
        assert_eq!(buf.average(), 4);

        buf.push(6); // Overwrites 3
        assert_eq!(buf.average(), 5);

        // Check iteration order (should be in insertion order: 4,5,6)
        let mut collected = [0; 3];
        let mut count = 0;

        for &val in buf.iter() {
            if count < 3 {
                collected[count] = val;
                count += 1;
            }
        }

        assert_eq!(count, 3);
        assert_eq!(collected, [4, 5, 6]);
    }

    #[test]
    fn test_dirty_flag_behavior() {
        let mut buf = SmoothBuffer::<5, f64>::new();

        // Fill buffer
        for i in 0..5 {
            buf.push(i as f64);
        }

        // First access should use cached values
        assert_eq!(buf.min(), 0.0);
        assert_eq!(buf.max(), 4.0);

        // Overwrite min
        buf.push(2.0); // Overwrites 0.0

        // Min should be recalculated
        assert_eq!(buf.min(), 1.0);

        // Overwrite multiple values including max
        buf.push(3.0); // Overwrites 1.0
        buf.push(2.0); // Overwrites 2.0
        buf.push(1.0); // Overwrites 3.0
        buf.push(0.5); // Overwrites 4.0 (max)

        // Max should be recalculated
        assert_eq!(buf.max(), 3.0);
    }

    #[test]
    fn test_pre_filled_edge_cases() {
        // Test with very large values
        let buf_large = SmoothBuffer::<5, f64>::pre_filled(1e10);
        assert!((buf_large.average() - 1e10).abs() < MARGIN);

        // Test with very small values
        let buf_small = SmoothBuffer::<5, f64>::pre_filled(1e-10);
        assert!((buf_small.average() - 1e-10).abs() < MARGIN);

        // Test with negative values
        let mut buf_neg = SmoothBuffer::<5, f64>::pre_filled(-5.0);
        assert!((buf_neg.average() - (-5.0)).abs() < MARGIN);
        assert!((buf_neg.min() - (-5.0)).abs() < MARGIN);
        assert!((buf_neg.max() - (-5.0)).abs() < MARGIN);
    }

    #[test]
    fn test_single_element_buffer() {
        let mut buf = SmoothBuffer::<1, i32>::new();

        // Push and check
        buf.push(42);
        assert_eq!(buf.average(), 42);
        assert_eq!(buf.min(), 42);
        assert_eq!(buf.max(), 42);

        // Overwrite and check
        buf.push(17);
        assert_eq!(buf.average(), 17);
        assert_eq!(buf.min(), 17);
        assert_eq!(buf.max(), 17);

        // Check iteration
        let mut has_value = false;

        for &val in buf.iter() {
            assert_eq!(val, 17);
            has_value = true;
        }

        assert!(has_value);
    }

    #[test]
    #[should_panic(expected = "Capacity must be larger than zero")]
    fn test_zero_capacity() {
        let _buf = SmoothBuffer::<0, f32>::new();
        // This should panic with the message in the attribute
    }

    #[test]
    fn test_gaussian_filter_with_spikes() {
        let mut buf = SmoothBuffer::<10, f64>::new();

        // Fill with a baseline value
        for _ in 0..8 {
            buf.push(1.0);
        }

        // Add a spike
        buf.push(10.0);
        buf.push(1.0);

        // Gaussian filter should reduce the impact of the spike compared to simple average
        let avg = buf.average();
        let gaussian = buf.gaussian_filter();

        // The gaussian value should be closer to 1.0 than the average
        assert!(gaussian < avg);
        assert!(gaussian > 1.0);

        // Another test case: values on both extremes
        let mut buf2 = SmoothBuffer::<5, f64>::new();
        buf2.push(1.0);
        buf2.push(10.0);
        buf2.push(5.0);
        buf2.push(5.0);
        buf2.push(1.0);

        // The gaussian filter should be weighted toward the center
        let gaussian2 = buf2.gaussian_filter();
        assert!(gaussian2 > 4.0); // average is 4.4
        assert!(gaussian2 < 5.5); // weighted toward center values
    }
}
