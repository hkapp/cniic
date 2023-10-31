use super::Point;
use std::collections::HashSet;
use std::thread;
use std::sync::Barrier;
use std::sync::atomic::{self, AtomicBool};
use std::sync::mpsc::{self, Sender, Receiver};

/*********************/
/* K-Means algorithm */
/*********************/

#[derive(Debug)]
pub struct Clusters<T> {
    centroids:       Vec<T>,
    assignment:      Vec<Vec<T>>,
    neighbours:      Vec<Vec<(usize, f64)>>, // ordered
}

pub fn cluster<I, T>(points: &mut I, nclusters: usize) -> Clusters<T>
    where I: Iterator<Item = T>,
        T: Point
{
    let avail_threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let nthreads = avail_threads.min(nclusters);

    let clusters_per_thread = distribute_clusters(nthreads, nclusters);
    println!("Clusters per thread: {:?}", &clusters_per_thread);

    let shared_state = SharedState::new(nthreads);

    let (senders, mut receivers): (Vec<_>, Vec<_>) = (0..nthreads)
                                                        .map(|_| mpsc::channel())
                                                        .unzip();
    // We're going to pop() the receivers, which takes from the bottom
    // So reverse the vec now
    receivers.reverse();

    let mut workers = Vec::new();
    for thread_id in 0..nthreads {
        let w = WorkerThread::new(thread_id, &clusters_per_thread, receivers.pop().unwrap(), senders.clone());
        workers.push(
            thread::spawn(||
                w.start(&shared_state)));
    }

    workers.into_iter()
        .for_each(|w| w.join().unwrap());
}

// Returns how many clusters for each thread, as evenly distributed as possible
fn distribute_clusters(nthreads: usize, nclusters: usize) -> Vec<usize> {
    let mut distribution = Vec::with_capacity(nthreads);
    let mut rem_clusters = nclusters;

    for i in 0..nthreads {
        let rem_threads = nthreads - i;
        let assigned = rem_clusters / rem_threads;
        distribution.push(assigned);
        rem_clusters -= assigned;
    }

    assert_eq!({let s: usize = distribution.iter().sum(); s}, nclusters);
    return distribution;
}

/*****************/
/* Worker thread */
/*****************/

type ThreadId = usize;

const ATOMIC_ORDERING: atomic::Ordering = atomic::Ordering::Relaxed;

type Message<T> = Result<(ThreadId, RemoteAssignment<T>), ThreadId>;

struct WorkerThread<T> {
    thread_id:     ThreadId,
    inward:        Receiver<Message<T>>,
    outwards:      Vec<Sender<Message<T>>>,
    my_assignment: WorkerAssignment<T>,
}

impl<T> WorkerThread<T> {
    fn new(thread_id: ThreadId, clusters_per_thread: &[usize], inward: Receiver<Message<T>>, outwards: Vec<Sender<Message<T>>>) -> Self {
        assert_eq!(clusters_per_thread.len(), outwards.len());
        WorkerThread {
            thread_id,
            inward,
            outwards,
            my_assignment: WorkerAssignment::new(thread_id, clusters_per_thread),
        }
    }

    fn send_remote_assignments(&mut self) {
        for (dest, data) in self.my_assignment.extract_remote_assignments() {
            let message = Ok((self.thread_id, data));
            self.send_message(dest, message);
        }
        self.notify_others_done_with_assign();
    }

    fn notify_others_done_with_assign(&self) {
        for channel in self.outwards {
            let message = Err(self.thread_id); // means EOF
            channel.send(message).unwrap()
        }
    }

    fn send_message(&self, destination: ThreadId, message: Message<T>) {
        self.outwards[destination]
            .send(message)
            .unwrap()  // we can't really continue if the message sending failed
    }

    // Keep reading the receiver channel until all other threads have sent EOF
    fn receive_remote_assignments(&mut self, nthreads: usize) -> bool {
        let mut not_received = HashSet::<_>::from_iter((0..nthreads).into_iter());
        not_received.remove(&self.thread_id);

        let mut received_any_changes = false;

        while !not_received.is_empty() {
            match self.inward.recv() {
                Ok(Ok((sender_id, remote_assignment))) => {
                    self.my_assignment.integrate_remote(sender_id, remote_assignment);
                    received_any_changes = true;
                },
                Ok(Err(thread_eof)) => {
                    let was_there = not_received.remove(&thread_eof);
                    assert!(was_there, "{}", thread_eof);
                }
                Err(_) => {
                    /* From https://doc.rust-lang.org/std/sync/mpsc/struct.RecvError.html:
                     *   The recv operation can only fail if the sending half of a channel (or sync_channel) is disconnected,
                     *   implying that no further messages will ever be received.
                     * In our case, this should never happen. Panic.
                     */
                     panic!("The channel for thread {} was shut", self.thread_id);
                }
            }
        }

        return received_any_changes;
    }
}

impl<T: Point> WorkerThread<T> {
    fn start(self, shared_state: &SharedState) -> Self {
        self.init_centroids();
        self.init_centroid_neighbours();

        let mut i_have_changes = true;
        while shared_state.consensus_continue(i_have_changes) {
            self.update_centroid_neighbours();
            let local_changes = self.assign_points_to_clusters(shared_state);
            self.send_remote_assignments();
            // This effectively acts like a sync barrier
            let remote_changes = self.receive_remote_assignments(shared_state.nthreads);
            i_have_changes = local_changes || remote_changes;
            self.compute_centroids();
        }

        return self;
    }

    fn assign_points_to_clusters(&mut self, shared_state: &SharedState) -> bool {
        assign_points_seq(&mut self.my_assignment, shared_state)
    }
}

/* WorkerAssignment */
/* The subset of the total point-to-cluster assignment that this worker thread knows about.
 * Before assigning points to clusters, these are all the points assigned to the clusters this worker is responsible for.
 * After assigning points to clusters, these are the assignments for the points previously in a clusters this worker is responsible for.
 *
 * This is a partitioned assignment table.
 * The first level is partitioned by thread.
 * The second level is partitioned by cluster.
 * Note that one of the first levels is for the current thread.
 */
struct WorkerAssignment<T> {
    thread_assignments: Vec<RemoteAssignment<T>>,
    current_thread:     ThreadId
}

impl<T> WorkerAssignment<T> {
    fn new(this_thread: ThreadId, clusters_per_thread: &[usize]) -> Self {
        WorkerAssignment {
            thread_assignments: clusters_per_thread
                                    .iter()
                                    .map(|nclusters| RemoteAssignment::new(*nclusters))
                                    .collect(),
            current_thread: this_thread,
        }
    }

    // Extract and return the non-empty thread-level assignments for other worker threads
    fn extract_remote_assignments<'a>(&'a mut self) -> impl Iterator<Item=(ThreadId, RemoteAssignment<T>)> + 'a {
        let this_thread = self.current_thread;

        self.thread_iter_mut()
            .filter(move |(that_thread, _)| this_thread != *that_thread)  // keep only remote assignments
            .filter(|(_, asg)| !asg.is_empty())  // nothing to do for empty asignments
            .map(|(other_tid, other_asg)| (other_tid, other_asg.extract()))
    }

    // Extract and return the assignment for the current worker thread
    fn extract_local_assignment(&mut self) -> (ThreadId, RemoteAssignment<T>) {
        let this_thread = self.current_thread;
        let local_asg = self.thread_assignments[this_thread].extract();
        return (this_thread, local_asg);
    }

    fn thread_iter_mut(&mut self) -> impl Iterator<Item=(ThreadId, &mut RemoteAssignment<T>)> {
        self.thread_assignments
            .iter_mut()
            .enumerate()
    }

    fn integrate_remote(&mut self, sender: ThreadId, remote_assignment: RemoteAssignment<T>) {
        self.thread_assignments[sender]
            .merge(remote_assignment)
    }

    fn is_empty(&self) -> bool {
        self.thread_assignments
            .iter()
            .all(|asg| asg.is_empty())
    }

    fn assign(&mut self, cluster_id: FullCentroidId, x: T) {
        let (target_thread, cluster_in_thread) = cluster_id;
        self.thread_assignments[target_thread]
            .assign(cluster_in_thread, x)
    }
}

type FullCentroidId    = (ThreadId, PartialCentroidId);
type PartialCentroidId = usize;

/* RemoteAssignment */
// Subset of a WorkerAssignment corresponding to a given thread.
struct RemoteAssignment<T> (Vec<Vec<T>>);

impl<T> RemoteAssignment<T> {
    fn new(nclusters: usize) -> Self {
        let assignments = (0..nclusters).map(|_| Vec::new()).collect();
        RemoteAssignment(assignments)
    }

    fn is_empty(&self) -> bool {
        self.0
            .iter()
            .all(|cluster_asg| cluster_asg.is_empty())
    }

    fn extract(&mut self) -> Self {
        let new_empty = RemoteAssignment::new(self.0.len());
        std::mem::replace(self, new_empty)
    }

    fn merge(&mut self, other: Self) {
        self.0.iter_mut()
            .zip(other.0.iter_mut())
            .for_each(|(v, w)| v.append(w))
    }

    fn assign(&mut self, target_cluster: PartialCentroidId, x: T) {
        self.0[target_cluster].push(x)
    }
}

/* point assignment */

// Returns true if some points changed assignment, false otherwise
// TODO share with the sequential code
fn assign_points_seq<T: Point>(curr_asg: &mut WorkerAssignment<T>, shared_state: &SharedState) -> bool {
    // 1. Extract the local points
    //    This must be the only non-empty section of the assignment
    let old_assignment = curr_asg.extract_local_assignment();
    assert!(curr_asg.is_empty());
    let new_assignment = curr_asg;

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
        let current_cluster = shared_state.get_centroid(cci);
        let mut min_dist = current_cluster.dist(&x);
        let mut closest_idx = Some(cci);

        // Check if we can early stop
        // TODO remove this 'if', it should be duplicate of the c_to_c_dist test below
        if min_dist <= current_cluster.certainty_radius {
            obvious_stay_count += 1;
        }
        else {
            // TODO: explain this
            let cluster_cutoff = 2.0 * min_dist;

            // Go over the neighbouring centroids in distance order
            for (tsi, c_to_c_dist) in current_cluster.neighbours.iter() {
                if *c_to_c_dist > cluster_cutoff {
                    neighbour_cutoff_count += 1;
                    break;
                }

                let t_centroid = shared_state.get_centroid(*tsi);
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
        new_assignment.assign(x, j);

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

/***************/
/* SharedState */
/***************/

struct SharedState {
    nthreads:           usize,
    anyone_has_changes: AtomicBool,
    barrier:            Barrier,
}

impl SharedState {
    fn new(nthreads: usize) -> Self {
        SharedState {
            nthreads,
            // WARNING: this starting value is very important
            // See consensus_continue()
            anyone_has_changes: AtomicBool::new(false),
            barrier:            Barrier::new(nthreads),
        }
    }

    // If any thread has changes, this returns true
    fn consensus_continue(&self, i_have_changes: bool) -> bool {
        if i_have_changes {
            self.anyone_has_changes.store(true, ATOMIC_ORDERING);
        }

        // Wait for every thread to get to this point
        let barrier_result = self.barrier.wait();

        let should_continue = self.anyone_has_changes.load(ATOMIC_ORDERING);
        if should_continue && barrier_result.is_leader() {
            // Reset the "has_any_changes" for the next round
            // We only need one thread to do that, hence the use of "is_leader()"
            // WARNING: technically the leader thread could starve until another thread gets back
            // to this consensus function
            self.anyone_has_changes.store(false, ATOMIC_ORDERING)
        }

        return should_continue;
    }
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

    #[test]
    fn radii() {
        let data = [(0, 0), (1, 0)];
        let clusters = super::cluster(&mut data.into_iter(), data.len());
        // Note: the certainty radius is half the distance to the closest centroid
        assert_eq!(clusters.certainty_radius(0), 0.5);
        assert_eq!(clusters.certainty_radius(1), 0.5);
    }
}
