mod seq;
#[allow(dead_code)]
mod parl;

pub trait Point: Sized {
    fn dist(&self, other: &Self) -> f64;

    // TODO have this return Option, s.t. it can return None when the input is empty
    fn mean(points: &[Self]) -> Self;
}

pub fn cluster<I, T>(points: &mut I, nclusters: usize) -> Clusters<T>
    where I: Iterator<Item = T>,
        T: Point
{
    Clusters::Seq(seq::cluster(points, nclusters))
}

pub enum Clusters<T> {
    Seq(seq::Clusters<T>)
}

impl<T> Clusters<T> {
    pub fn into_iter(self) -> impl Iterator<Item=(T, Vec<T>)> {
        match self {
            Clusters::Seq(seq) => seq.into_iter(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::collections::HashSet;

    impl Point for (i32, i32) {
        fn dist(&self, other: &Self) -> f64 {
            fn diff_squared(a: i32, b: i32) -> f64 {
                ((a - b) as f64).powi(2)
            }

            (diff_squared(self.0, other.0) + diff_squared(self.1, other.1)).sqrt()
        }

        fn mean(points: &[Self]) -> Self {
            let sum = points.iter()
                        .map(|(a, b)| (*a as i64, *b as i64))
                        .fold((0, 0), |(r, u), (a, b)| (r + a, u + b));

            let div = |x: i64| -> i32 {
                (x / points.len() as i64) as i32
            };
            (div(sum.0), div(sum.1))
        }
    }

    /* . . .         * * *
     * . * .    =>   * * *
     * . . .         * * *
     */
    pub fn square_centered_at(p: (i32, i32)) -> Vec<(i32, i32)> {
        let mut square = Vec::new();
        for i in -1..2 {
            for j in -1..2 {
                square.push((p.0 + i, p.1 + j));
            }
        }
        square
    }

    #[test]
    fn dist1() {
        assert_eq!((0, 0).dist(&(0, 1)), 1.0);
    }

    #[test]
    fn dist2() {
        let sq_center = (-100, 0);
        let correct_centroid = (-11, 0);
        let wrong_centroid = (11, 0);

        for p in square_centered_at(sq_center) {
            let closer = correct_centroid.dist(&p);
            let further = wrong_centroid.dist(&p);
            assert!(closer < further);
        }
    }

    #[test]
    fn mean1() {
        let sq_center = (-100, 0);
        assert_eq!(Point::mean(&square_centered_at(sq_center)), sq_center);
    }

}
