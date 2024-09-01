#![warn(clippy::std_instead_of_core, clippy::std_instead_of_alloc)]
#![no_std]
use core::slice::{Iter, IterMut};
use libm::sinf;

/// Simple fixed size ringbuffer with fast averaging and smoothing.
/// Do not use if the order of retrieval of each element matters.

pub struct SmoothBuffer<const CAP: usize> {
    data: [f32; CAP],
    head: usize,
    sum: Option<f32>,
    max: Option<f32>,
    min: Option<f32>,
    filled_len: usize,
}

impl<const CAP: usize> Default for SmoothBuffer<CAP> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAP: usize> SmoothBuffer<CAP> {
    pub fn new() -> Self {
        SmoothBuffer {
            data: [0.0; CAP],
            head: 0,
            sum: None,
            max: None,
            min: None,
            filled_len: 0,
        }
    }

    /// Fast! Sum is always kept up to date on push. No need iterate.
    pub fn average(&self) -> f32 {
        if self.filled_len > 0 {
            return self.sum.unwrap_or(0.0) / self.filled_len as f32;
        }
        0.0
    }

    pub fn clear(&mut self) {
        // for n in 0..self.data.len(){
        //     self.data[n] = 0.0;
        // }
        self.sum = None;
        self.max = None;
        self.min = None;
        self.filled_len = 0;
        self.head = 0;
    }

    pub fn is_empty(&self) -> bool {
        self.filled_len == 0
    }

    pub fn max(&self) -> f32 {
        self.max.unwrap_or(0.0)
    }

    pub fn min(&self) -> f32 {
        self.min.unwrap_or(0.0)
    }

    pub fn capacity(&self) -> usize {
        CAP
    }

    pub fn len(&self) -> usize {
        self.filled_len
    }

    pub fn push(&mut self, value: f32) {
        match self.max {
            None => self.max = Some(value),
            Some(max) => self.max = Some(f32::max(max, value)),
        }
        match self.min {
            None => self.min = Some(value),
            Some(min) => self.min = Some(f32::min(min, value)),
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

    pub fn iter(&self) -> Iter<f32> {
        self.data[0..self.filled_len].iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<f32> {
        self.data[0..self.filled_len].iter_mut()
    }

    /// Will perform super fast hard-coded weighted average with up to 5 elements.
    /// Above that will use a sine-based formula, which is slower.
    /// TODO: Expand number of hard coded weights.
    pub fn smooth(&self) -> f32 {
        let len = self.filled_len;
        let mut sum = 0.0;
        match len {
            0 => {}
            1 => sum = self.data[0],
            2 => sum = (self.data[0] + self.data[1]) * 0.5,
            3 => {
                let weights = [0.025, 0.95, 0.025];
                weights
                    .iter()
                    .enumerate()
                    .for_each(|(i, w)| sum += self.data[i] * w);
            }
            4 => {
                let weights = [0.015, 0.485, 0.485, 0.015];
                weights
                    .iter()
                    .enumerate()
                    .for_each(|(i, w)| sum += self.data[i] * w);
            }
            5 => {
                let weights = [0.016667, 0.0333, 0.900066, 0.0333, 0.016667];
                weights
                    .iter()
                    .enumerate()
                    .for_each(|(i, w)| sum += self.data[i] * w);
            }
            _ => {
                use core::f32::consts::TAU;
                for i in 0..self.len() {
                    let x = i as f32 / self.filled_len as f32;
                    let y = (sinf((x - 0.25) * TAU) / 2.0) + 0.5;
                    sum += self.data[i] * y;
                }
                sum /= len as f32;
                sum *= 2.0;
            }
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_push() {
        const CAP: usize = 10;
        let mut buf = SmoothBuffer::<CAP>::new();
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
        let mut buf = SmoothBuffer::<10>::new();
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
        let mut buf = SmoothBuffer::<10>::new();
        let len = 7;
        for n in 0..len {
            buf.push(n as f32);
        }

        for (i, value) in buf.iter().enumerate() {
            assert_eq!(i as f32, *value);
        }

        assert!(buf.iter().len() == len);
    }

    #[test]
    fn smoothing() {
        let mut buf = SmoothBuffer::<10>::new();
        let values = [
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        ];
        for v in values {
            buf.push(v);
        }
        // Smoothed value won't be exactly the same! Will be correct to a few decimal places though
        assert!(buf.smooth() - 3.0 < 0.000001);
    }
}
