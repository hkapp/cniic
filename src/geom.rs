use image::Rgb;

// We understand here "Euclidean distance"
pub trait Distance {
    fn dist(&self, other: &Self) -> f64;
}

impl Distance for Rgb<u8> {
    fn dist(&self, other: &Self) -> f64 {
        //self.0[..]
            //.iter()
            //.zip(other.0[..].iter())
            //.map(|(x, y)| (*x as f64 - *y as f64).powi(2))
            //.sum::<f64>()
            //.sqrt()
        let f = |i: usize| {
            let x = self[i];
            let y = other[i];
            let diff = x as i32 - y as i32;
            (diff * diff) as f64
        };
        (f(0) + f(1) + f(2)).sqrt()
    }
}
