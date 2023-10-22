pub trait Point: Sized {
    fn dist(&self, other: &Self) -> f64;

    fn mean(points: &[Self]) -> Self;
}

#[derive(Debug)]
pub struct Clusters<T> {
    centroids:  Vec<T>,
    assignment: Vec<Vec<T>>
}

/* K-Means algorithm */

pub fn cluster<I, T>(points: &mut I, nclusters: usize) -> Clusters<T>
    where I: Iterator<Item = T>,
        T: Point
{
    let mut clusters = init(points, nclusters);
    let mut changed_assignment = true;

    let mut i = 0;
    while changed_assignment {
        changed_assignment = assign_points(&mut clusters);
        // WARNING: always compute the centroids: the caller can use them
        compute_centroids(&mut clusters);
        i += 1;
    }
    println!("#iterations: {}", i);

    return clusters;
}

type Assignment<T> = Vec<Vec<T>>;

// Consider requiring ExactSizeIterator
fn init_assignment<T, I>(points: &mut I, nclusters: usize) -> Assignment<T>
    where I: Iterator<Item = T>,
        T: Point
{
    let mut init_assignment = Vec::with_capacity(nclusters);
    for _ in 0..nclusters {
        init_assignment.push(Vec::new());
    }

    for (i, x) in points.enumerate() {
        let vec = init_assignment.get_mut(i % nclusters).unwrap();
        vec.push(x);
    }

    return init_assignment;
}

fn init<T, I>(points: &mut I, nclusters: usize) -> Clusters<T>
    where I: Iterator<Item = T>,
        T: Point
{
    let assignment = init_assignment(points, nclusters);

    let mut clusters = Clusters {
        assignment,
        centroids: Vec::with_capacity(nclusters),
    };

    compute_centroids(&mut clusters);

    return clusters;
}

fn compute_centroids<T: Point>(clusters: &mut Clusters<T>) {
    // Note: Vec::clear() keeps the underlying allocated buffer, which is what we want
    clusters.centroids.clear();
    for points_in_cluster in clusters.assignment.iter() {
        clusters.centroids.push(T::mean(&points_in_cluster));
    }
}

// Returns true if some points changed assignment, false otherwise
fn assign_points<T: Point>(clusters: &mut Clusters<T>) -> bool {
    // This is the dummiest method: full join

    // 1. Extract all the vectors of points
    let new_assignment = (0..clusters.len()).map(|_| Vec::new()).collect();
    let old_assignment = std::mem::replace(&mut clusters.assignment, new_assignment);

    // 2. Move the points accordingly
    let mut some_change = false;
    for (i, x) in old_assignment.into_iter()
                            .enumerate()
                            .flat_map(|(i, v)| v.into_iter()
                                                .map(move |x| (i, x)))
    {
        // Find the closest centroid
        // Note that we can't use Iterator::min_by_key() because f64 is only PartialOrd
        // https://stackoverflow.com/questions/69665188/min-max-of-vecf64-trait-ord-is-not-implemented-for-xy
        let (j, _) = clusters.centroids
                        .iter_mut()
                        .enumerate()
                        .map(|(i, c)| (i, c.dist(&x)))
                        .min_by(|(i, d1), (j, d2)|
                            d1.partial_cmp(d2).unwrap_or_else(|| i.cmp(j)))
                        .unwrap();

        clusters.assignment[j].push(x);

        if i != j {
            some_change = true;
        }
    }

    return some_change;
}

/* Public API for Clusters */

impl<T> Clusters<T> {
    // Note: we don't implement the IntoIterator trait because we'd need
    //       to define our own iterator type
    pub fn into_iter(self) -> impl Iterator<Item=(T, Vec<T>)> {
        self.centroids
            .into_iter()
            .zip(self.assignment
                    .into_iter())
    }

    fn len(&self) -> usize {
        assert_eq!(self.centroids.len(), self.assignment.len());
        self.centroids.len()
    }
}

/* Unit tests */

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

    fn assert_centroids<T>(clusters: &Clusters<T>, expected_centroids: &[T])
        where T: Clone + Eq + std::hash::Hash + std::fmt::Debug
    {
        assert_eq!(clusters.centroids.len(), expected_centroids.len());

        let centroids = HashSet::<_>::from_iter(clusters.centroids.iter().clone());
        println!("Set of centroids: {:?}", centroids);
        for ex in expected_centroids {
            assert!(centroids.contains(ex), "Not present: {:?}", &ex);
        }
    }

    #[test]
    fn all_clusters() {
        let data = [(0, 0), (1, 1)];
        let clusters = super::cluster(&mut data.into_iter(), data.len());
        assert_centroids(&clusters, &data[..]);

        for i in 0..data.len() {
            assert_eq!(clusters.assignment[i], vec![clusters.centroids[i]]);
        }
    }

    /* . . .         * * *
     * . * .    =>   * * *
     * . . .         * * *
     */
    fn square_centered_at(p: (i32, i32)) -> Vec<(i32, i32)> {
        let mut square = Vec::new();
        for i in -1..2 {
            for j in -1..2 {
                square.push((p.0 + i, p.1 + j));
            }
        }
        square
    }

    #[test]
    fn square1() {
        let data = square_centered_at((0, 0));
        let ndata = data.len();
        println!("{:?}", &data);
        let clusters = super::cluster(&mut data.into_iter(), 1);
        assert_centroids(&clusters, &[(0, 0)][..]);
        assert_eq!(clusters.assignment[0].len(), ndata);
    }

    #[test]
    fn squares2() {
        let sq_centers = [(-100, 0), (100, 0)];
        let data = sq_centers.into_iter()
                    .map(|p| square_centered_at(p))
                    .reduce(|mut v1, mut v2| {
                        v1.append(&mut v2);
                        v1
                    })
                    .unwrap();
        println!("{:?}", &data);

        let clusters = super::cluster(&mut data.into_iter(), 2);
        println!("{:?}", &clusters);
        assert_centroids(&clusters, &sq_centers[..]);
    }

    #[test]
    fn dist1() {
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
