use std::cell::Cell;
use std::cmp::{min, max};

pub trait Point: Sized {
    fn dist(&self, other: &Self) -> f64;

    // This must return None if the given slice is empty
    fn mean(points: &[Self]) -> Option<Self>;
}

pub struct Clusters<T> {
    centroids:       Vec<T>,
    assignment:      Vec<Vec<T>>,
    neighbours:      NeighbourGraph,
}

/*********************/
/* K-Means algorithm */
/*********************/

pub fn cluster<T: Point>(points: Vec<T>, nclusters: usize) -> Clusters<T> {
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

    // Sanity check
    check_enough_active_clusters(&clusters, nclusters);

    return clusters;
}

fn check_enough_active_clusters<T>(clusters: &Clusters<T>, requested_nclusters: usize) {
    let point_count = clusters.assignment
                            .iter()
                            .map(|v| v.len())
                            .sum();
    // We allow for 1% drift from the value requested by the client
    let min_cluster_count = std::cmp::min(point_count, (0.99 * requested_nclusters as f64) as usize);

    let active_cluster_count = clusters.assignment
                                    .iter()
                                    .filter(|v| v.len() > 0)
                                    .count();

    assert!(active_cluster_count >= min_cluster_count,
        "Not enough active clusters\nRequested {}, got {} (min allowed: {})",
        requested_nclusters, active_cluster_count, min_cluster_count);
}

type Assignment<T> = Vec<Vec<T>>;

fn init_assignment<T: Point>(mut points: Vec<T>, nclusters: usize) -> Assignment<T> {
    // Note: we could actually implement the same mechanism if the input is an iterator
    /* Each cluster gets one point
     * The last one gets all the remaining points
     */
    let mut init_assignment = Vec::new();
    for _ in 0..(nclusters-1) {
        init_assignment.push(points.pop().into_iter().collect());
    }
    init_assignment.push(points);

    return init_assignment;
}

fn init<T: Point>(points: Vec<T>, nclusters: usize) -> Clusters<T> {
    let assignment = init_assignment(points, nclusters);
    let centroids  = init_centroids(&assignment);
    let neighbours = init_neighbours(&centroids);

    Clusters {
        assignment,
        centroids,
        neighbours
    }
}

/*************/
/* centroids */
/*************/

fn init_centroids<T: Point>(assignment: &Assignment<T>) -> Vec<T> {
    let nclusters = assignment.len();
    let mut centroids = Vec::with_capacity(nclusters);
    compute_centroids(&assignment, &mut centroids);
    return centroids;
}

fn compute_centroids<T: Point>(assignment: &Assignment<T>, centroids: &mut Vec<T>) {
    for points_in_cluster in assignment.iter() {
        // FIXME need to cover the case when the cluster has no data
        centroids.push(T::mean(&points_in_cluster).unwrap());
    }
}

fn update_centroids<T: Point>(clusters: &mut Clusters<T>) {
    // Note: Vec::clear() keeps the underlying allocated buffer, which is what we want
    clusters.centroids.clear();
    compute_centroids(&clusters.assignment, &mut clusters.centroids);
}

/**************/
/* neighbours */
/**************/

// The neighbours of a single centroid
struct NeighbouringCentroids {
    watermark:  Cell<usize>,
    neighbours: Vec<(CentroidId, f64)>, // ordered by distance
}

impl NeighbouringCentroids {
    fn new(src: CentroidId, nclusters: usize) -> Self {
        let neighbours = (0..nclusters)  // generate neighbours
                            .filter(|dst| *dst != src)  // can't be self
                            .map(|dst| (dst, 0.0))
                            .collect::<Vec<_>>();

        let init_watermark = neighbours.len(); // this value is important for init_neighbours

        NeighbouringCentroids {
            neighbours: neighbours,
            watermark:  Cell::new(init_watermark),
        }
    }

    fn update_distances<T: Point>(&mut self, src: &T, all_centroids: &[T]) {
        for neigh in self.neighbours.iter_mut() {
            let dst = &all_centroids[neigh.0];
            neigh.1 = src.dist(dst);
            assert!(neigh.1.is_finite());
        }
    }

    fn sort(&mut self) {
        /*let already_sorted = self.neighbours
                                .windows(2)
                                // Optimization: consider it "good enough" if the first sqrt(nclusters) are sorted
                                // This leads to imprecision. Though when we get to the point where the first sqrt(nclusters)
                                // remain sorted, we can assume that the points don't go so far in the list of neighbours anymore
                                .take((centroids.len() as f64).sqrt() as usize)
                                .find(|xs| xs[0].1 > xs[1].1)
                                .is_none();
        //let already_sorted=false;
        if !already_sorted {*/
            // Optimization: use unstable sorting
            // We don't care about the stability of the ordering, we're interested in speed
            self.neighbours.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).ok_or_else(|| format!("{} <> {}", a, b)).unwrap());
        //}
    }

    fn shrink(&mut self, nclusters: usize) {
        let upper_bound = nclusters - 1;  // max possible num neighbours
        let lower_bound = nclusters.ilog2() as usize + 1;  // add 1 to simulate rounding up
        let dyn_val = 2 * self.watermark.get();
        let new_num_neighbours = min(
                                    max(
                                        dyn_val,
                                        lower_bound),
                                    upper_bound);

        // Just do some logging, out of curiosity
        if dyn_val < lower_bound {
            println!("NeighbouringCentroids::shrink: lower bound triggered");
            println!("                               dyn_val = {} < {} = lower_bound", dyn_val, lower_bound);
        }

        // TODO need to cover the case where the watermark value requires getting more neighbours
        assert!(new_num_neighbours <= self.neighbours.len());

        self.neighbours.truncate(new_num_neighbours);

        // Reset the watermark
        self.watermark.replace(0);
    }

    fn iter_and_record(&self) -> IterAndRecord {
        IterAndRecord {
            neigh_list: self,
            next_pos:  0
        }
    }

    fn certainty_radius(&self) -> f64 {
        self.neighbours.get(0).map(|x| x.1 / 2.0).unwrap_or(f64::INFINITY)
    }
}

struct IterAndRecord<'a> {
    neigh_list: &'a NeighbouringCentroids,
    next_pos:   usize,
}

impl<'a> Iterator for IterAndRecord<'a> {
    type Item = &'a (CentroidId, f64);

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.neigh_list.neighbours.get(self.next_pos);
        self.next_pos += 1;
        value
    }
}

impl<'a> Drop for IterAndRecord<'a> {
    fn drop(&mut self) {
        self.neigh_list.watermark.replace(self.next_pos - 1);
    }
}

type CentroidId = usize;
// All the neighbours, taken together, form a graph
// For correctness sake, we better hope that this graph is connected
type NeighbourGraph = Vec<NeighbouringCentroids>;

fn init_neighbours<T: Point>(centroids: &[T]) -> NeighbourGraph {
    let nclusters = centroids.len();
    let mut neighbours = (0..nclusters)  // for each clusters
                            .map(|i| NeighbouringCentroids::new(i, nclusters))
                            .collect();
    compute_neighbours(&centroids, &mut neighbours);
    return neighbours;
}

// Optimization: keep the array in the previous order
// There are good chances that the array is already sorted in this case
fn compute_neighbours<T: Point>(centroids: &[T], neigh_graph: &mut NeighbourGraph) {
    for (c, neighbours) in centroids.iter().zip(neigh_graph.iter_mut()) {
        neighbours.update_distances(c, centroids);
        // Note: this ordering of sort vs. shrink leads to more work, but also more accuracy
        // The extra work should only be an effect at the beginning
        neighbours.sort();
        // Optimization: dynamically adapt the number of neighbours for each cluster
        let nclusters = centroids.len();
        neighbours.shrink(nclusters);
    }
}

fn update_neighbours<T: Point>(clusters: &mut Clusters<T>) {
    // Optimization: don't clear the buffer and keep it in the same order
    compute_neighbours(&clusters.centroids, &mut clusters.neighbours);
}

/********************/
/* point assignment */
/********************/

// Returns true if some points changed assignment, false otherwise
fn assign_points<T: Point>(clusters: &mut Clusters<T>) -> bool {
    // 1. Extract all the vectors of points
    let new_assignment = (0..clusters.len()).map(|_| Vec::new()).collect();
    let old_assignment = std::mem::replace(&mut clusters.assignment, new_assignment);

    // 2. Move the points accordingly
    let mut some_change = false;
    let mut obvious_stay_count = 0;
    let mut neighbour_cutoff_count = 0;
    let mut moved_count = 0;
    let mut stayed_count = 0;
    let mut max_tested_neighbours = vec![0; old_assignment.len()];
    let mut sum_tested_neighbours = 0;
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

            let mut tested_neighbours = 0;
            // Go over the neighbouring centroids in distance order
            for (tsi, c_to_c_dist) in clusters.neighbours[cci].iter_and_record() {
                tested_neighbours += 1;

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
            max_tested_neighbours[cci] = std::cmp::max(max_tested_neighbours[cci], tested_neighbours);
            sum_tested_neighbours += tested_neighbours;
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
    println!("Global max tested neighbours: {}", max_tested_neighbours.iter().max().unwrap());
    println!("Min of max tested neighbours: {}", max_tested_neighbours.iter().min().unwrap());
    //println!("{:?}", max_tested_neighbours);
    //println!("{:?}", clusters.centroids);
    let tot_points = moved_count + stayed_count - obvious_stay_count;
    if tot_points != 0 {
        println!("Average tested neighbours: {}", sum_tested_neighbours / (moved_count + stayed_count - obvious_stay_count));
    }

    return some_change;
}

/***************************/
/* Public API for Clusters */
/***************************/

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
        self.neighbours[cid].certainty_radius()
    }
}

/**************/
/* Unit tests */
/**************/

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

        fn mean(points: &[Self]) -> Option<Self> {
            if points.len() == 0 {
                return None;
            }

            let sum = points.iter()
                        .map(|(a, b)| (*a as i64, *b as i64))
                        .fold((0, 0), |(r, u), (a, b)| (r + a, u + b));

            let div = |x: i64| -> i32 {
                (x / points.len() as i64) as i32
            };
            let res = (div(sum.0), div(sum.1));
            Some(res)
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
        let data = vec![(0, 0), (1, 1)];
        let clusters = super::cluster(data.clone(), data.len());
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
        println!("{:?}", &data);
        let clusters = super::cluster(data.clone(), 1);
        assert_centroids(&clusters, &[(0, 0)][..]);
        assert_eq!(clusters.assignment[0].len(), data.len());
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

        let clusters = super::cluster(data, 2);
        assert_centroids(&clusters, &sq_centers[..]);
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
        assert_eq!(Point::mean(&square_centered_at(sq_center)), Some(sq_center));
    }

    #[test]
    fn radii() {
        let data = vec![(0, 0), (1, 0)];
        let nclusters = data.len();
        let clusters = super::cluster(data, nclusters);
        // Note: the certainty radius is half the distance to the closest centroid
        assert_eq!(clusters.certainty_radius(0), 0.5);
        assert_eq!(clusters.certainty_radius(1), 0.5);
    }

    #[test]
    fn proper_init_asg() {
        let data = vec![(1000, 0), (1000, 1), (-1000, 0), (-1000, 1)];
        let _clusters = super::cluster(data, 3);
        // Note: the above should fail an assertion in the tested bug
    }
}
