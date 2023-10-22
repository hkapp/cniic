pub trait Point: Sized {
    fn dist(&self, other: &Self) -> f64;

    fn mean(points: &[Self]) -> Self;
}

#[derive(Debug)]
pub struct Cluster<T> {
    pub centroid: T,
    pub points:   Vec<T>
}

pub fn cluster<I, T>(points: &mut I, nclusters: usize) -> Vec<Cluster<T>>
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

// Consider requiring ExactSizeIterator
fn init<T, I>(points: &mut I, nclusters: usize) -> Vec<Cluster<T>>
    where I: Iterator<Item = T>,
        T: Point
{
    // 1. Assign the points to the various clusters

    let mut init_assignment = Vec::with_capacity(nclusters);
    for _ in 0..nclusters {
        init_assignment.push(Vec::new());
    }

    for (i, x) in points.enumerate() {
        let vec = init_assignment.get_mut(i % nclusters).unwrap();
        vec.push(x);
    }

    // 2. Build the clusters by running mean() a first time
    // Note: we can avoid this if we require Clone on the type T

    let mut clusters = Vec::with_capacity(nclusters);
    for c_elems in init_assignment.into_iter() {
        let new_c = Cluster {
            centroid: T::mean(&c_elems),
            points:   c_elems,
        };

        clusters.push(new_c);
    }

    return clusters;
}

fn compute_centroids<T: Point>(clusters: &mut Vec<Cluster<T>>) {
    for c in clusters.iter_mut() {
        c.centroid = T::mean(&c.points);
    }
}

// Returns true if some points changed assignment, false otherwise
fn assign_points<T: Point>(clusters: &mut Vec<Cluster<T>>) -> bool {
    // This is the dummiest method: full join

    // 1. Extract all the vectors of points
    let mut old_assignment = Vec::with_capacity(clusters.len());
    for c in clusters.iter_mut() {
        let old_points = std::mem::replace(&mut c.points, Vec::new());
        old_assignment.push(old_points);
        //old_assignment.push(c.points);
        //c.points = Vec::new();
    }

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
        let (j, _, c_assign) = clusters.iter_mut()
                                .enumerate()
                                .map(|(i, c)| (i, c.centroid.dist(&x), c))
                                .min_by(|(i, d1, _), (j, d2, _)|
                                    d1.partial_cmp(d2).unwrap_or_else(|| i.cmp(j)))
                                .unwrap();

        c_assign.points.push(x);

        if i != j {
            some_change = true;
        }
    }

    return some_change;
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

    fn assert_centroids<T>(clusters: &[Cluster<T>], expected_centroids: &[T])
        where T: Clone + Eq + std::hash::Hash + std::fmt::Debug
    {
        assert_eq!(clusters.len(), expected_centroids.len());

        let centroids = HashSet::<_>::from_iter(clusters.iter().map(|c| c.centroid.clone()));
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

        for c in clusters {
            assert_eq!(c.points, vec![c.centroid]);
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
        assert_eq!(clusters[0].points.len(), ndata);
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
