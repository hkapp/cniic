use super::Point;

#[derive(Debug)]
pub struct Clusters<T> {
    centroids:       Vec<T>,
    assignment:      Vec<Vec<T>>,
    neighbours:      Vec<Vec<(usize, f64)>>, // ordered
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
        update_centroids(&mut clusters);
        update_neighbours(&mut clusters);
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
    let centroids  = init_centroids(&assignment);
    let neighbours = init_neighbours(&centroids);

    Clusters {
        assignment,
        centroids,
        neighbours
    }
}

/* centroids */

fn init_centroids<T: Point>(assignment: &Assignment<T>) -> Vec<T> {
    let nclusters = assignment.len();
    let mut centroids = Vec::with_capacity(nclusters);
    compute_centroids(&assignment, &mut centroids);
    return centroids;
}

fn compute_centroids<T: Point>(assignment: &Assignment<T>, centroids: &mut Vec<T>) {
    for points_in_cluster in assignment.iter() {
        centroids.push(T::mean(&points_in_cluster));
    }
}

fn update_centroids<T: Point>(clusters: &mut Clusters<T>) {
    // Note: Vec::clear() keeps the underlying allocated buffer, which is what we want
    clusters.centroids.clear();
    compute_centroids(&clusters.assignment, &mut clusters.centroids);
}

/* neighbours */

type Neighbours = Vec<Vec<(usize, f64)>>;

fn init_neighbours<T: Point>(centroids: &[T]) -> Neighbours {
    let nclusters = centroids.len();
    let mut neighbours = (0..nclusters)  // for each clusters
                            .map(|i|
                                (0..nclusters)  // generate neighbours
                                .filter(|j| *j != i)  // can't be self
                                .map(|j| (j, 0.0))
                                .collect())
                            .collect();
    compute_neighbours(&centroids, &mut neighbours);
    return neighbours;
}

// Optimization: keep the array in the previous order
// There are good chances that the array is already sorted in this case
fn compute_neighbours<T: Point>(centroids: &[T], neighbours: &mut Neighbours) {
    let mut sort_count = 0;
    for (i, c) in centroids.iter().enumerate() {
        for neigh in neighbours[i].iter_mut() {
            neigh.1 = c.dist(&centroids[neigh.0]);
            assert!(neigh.1.is_finite());
        }

        let already_sorted = neighbours[i]
                                .windows(2)
                                // Optimization: consider it "good enough" if the first sqrt(nclusters) are sorted
                                // This leads to imprecision. Though when we get to the point where the first sqrt(nclusters)
                                // remain sorted, we can assume that the points don't go so far in the list of neighbours anymore
                                .take((centroids.len() as f64).sqrt() as usize)
                                .find(|xs| xs[0].1 > xs[1].1)
                                .is_none();
        //let already_sorted=false;
        if !already_sorted {
            sort_count += 1;
            // Optimization: use unstable sorting
            // We don't care about the stability of the ordering, we're interested in speed
            neighbours[i].sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).ok_or_else(|| format!("{} <> {}", a, b)).unwrap());
        }
    }
    println!("Sorted {} times", sort_count);
}

fn update_neighbours<T: Point>(clusters: &mut Clusters<T>) {
    // Optimization: don't clear the buffer and keep it in the same order
    compute_neighbours(&clusters.centroids, &mut clusters.neighbours);
}

/* point assignment */

// Returns true if some points changed assignment, false otherwise
fn assign_points<T: Point>(clusters: &mut Clusters<T>) -> bool {
    // This is the dummiest method: full join

    // 1. Extract all the vectors of points
    let new_assignment = (0..clusters.len()).map(|_| Vec::new()).collect();
    let old_assignment = std::mem::replace(&mut clusters.assignment, new_assignment);

    // 2. Move the points accordingly
    let mut some_change = false;
    let mut obvious_stay_count = 0;
    let mut neighbour_cutoff_count = 0;
    let mut moved_count = 0;
    let mut stayed_count = 0;
    for (cci, x) in old_assignment.into_iter()
                            .enumerate()
                            .flat_map(|(i, v)| v.into_iter()
                                                .map(move |x| (i, x)))
    {
        // Start with the current centroid
        let current_cluster = &clusters.centroids[cci];
        let mut min_dist = current_cluster.dist(&x);
        let mut closest_idx = Some(cci);

        // Check if we can early stop
        // TODO remove this 'if', it should be duplicate of the c_to_c_dist test below
        if min_dist <= clusters.certainty_radius(cci) {
            obvious_stay_count += 1;
        }
        else {
            // TODO: explain this
            let cluster_cutoff = 2.0 * min_dist;

            // Go over the neighbouring centroids in distance order
            for (tsi, c_to_c_dist) in clusters.neighbours[cci].iter() {
                if *c_to_c_dist > cluster_cutoff {
                    neighbour_cutoff_count += 1;
                    break;
                }

                let t_centroid = &clusters.centroids[*tsi];
                let t_dist = t_centroid.dist(&x);

                if t_dist < min_dist {
                    min_dist = t_dist;
                    closest_idx = Some(*tsi);
                }

                // This codepath almost never triggers but actually has a quite high impact on performance
                /*
                // Check if we can early stop
                if t_dist <= clusters.certainty_radius(*tsi) {
                    if closest_idx != Some(*tsi) {
                        // The only case that seems to make sense is when the distance is equal
                        assert_eq!(t_dist, min_dist, "{:?}", &clusters.neighbours[*tsi]);
                        closest_idx = Some(*tsi);
                    }
                    cerainty_radius_count += 1;
                    break;
                }*/
            }
        }

        let j = closest_idx.unwrap();
        clusters.assignment[j].push(x);

        if cci != j {
            some_change = true;
            moved_count += 1;
        }
        else {
            stayed_count += 1;
        }
    }

    println!("Moved: {}, Stayed: {}", moved_count, stayed_count);
    println!("Stopped early: {}", obvious_stay_count + neighbour_cutoff_count);
    println!("Because of");
    println!("..certainty radius of its previous centroid: {}", obvious_stay_count);
    println!("..neighbour cutoff: {}", neighbour_cutoff_count);

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

    fn certainty_radius(&self, cid: usize) -> f64 {
        self.neighbours[cid].get(0).map(|x| x.1).unwrap_or(0.0) / 2.0
    }
}

/* Unit tests */

#[cfg(test)]
mod test {
    use super::*;
    use super::super::test::square_centered_at;
    use std::collections::HashSet;

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
    fn radii() {
        let data = [(0, 0), (1, 0)];
        let clusters = super::cluster(&mut data.into_iter(), data.len());
        // Note: the certainty radius is half the distance to the closest centroid
        assert_eq!(clusters.certainty_radius(0), 0.5);
        assert_eq!(clusters.certainty_radius(1), 0.5);
    }
}
