pub trait Point: Sized {
    fn dist(&self, other: &Self) -> f64;

    fn mean(points: &[Self]) -> Self;
}

pub struct Cluster<T> {
    pub centroid: T,
    pub points:   Vec<T>
}

pub fn cluster<I, T>(points: &mut I, nclusters: usize) -> Vec<Cluster<T>>
    where I: Iterator<Item = T>,
        T: Point
{
    let mut clusters = init(points, nclusters);
    let mut changed_assignment = false;

    while changed_assignment {
        compute_centroids(&mut clusters);
        changed_assignment = assign_points(&mut clusters);
    }

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
        // Note that we can't use Iterator::min_by() because f64 is only PartialOrd
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
