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
