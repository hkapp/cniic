use super::Point;
use std::collections::HashSet;
use std::ops::Index;
use std::thread;
use std::sync::{Arc, Barrier, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::sync::atomic::{self, AtomicBool};
use std::sync::mpsc::{self, Sender, Receiver};

/*********************/
/* K-Means algorithm */
/*********************/

pub fn cluster<I, T>(points: &mut I, nclusters: usize) -> Clusters<T>
    where I: Iterator<Item = T>,
        T: Point + Send + Sync + 'static
{
    // Figure out how many threads to use
    let avail_threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let nthreads = avail_threads.min(nclusters);
    println!("Using {} threads", nthreads);

    // Split the clusters per threads
    let clusters_per_thread = distribute_clusters(nthreads, nclusters);
    println!("Clusters per thread: {:?}", &clusters_per_thread);

    // Split the data per thread
    let assignments = init_assignments(points, &clusters_per_thread);

    // Initialize the centroids
    // WARNING: we don't initialize the neighbours yet. Each thread will do this
    let centroids: Vec<_> = assignments.iter()
                                .map(|asg| init_centroids(&asg))
                                .collect();
    //self.init_centroid_neighbours();

    let shared_state = SharedState::new(nthreads, centroids);
    let shared_state = Arc::new(shared_state);

    let (senders, receivers): (Vec<_>, Vec<_>) = (0..nthreads)
                                                        .map(|_| mpsc::channel())
                                                        .unzip();

    let mut workers = Vec::new();
    let mut all_assignments = assignments.into_iter();
    let mut receivers_iter = receivers.into_iter();

    for thread_id in 0..nthreads {
        let local_assignment = all_assignments.next().unwrap();
        let messages_send = senders.clone();
        let messages_receive = receivers_iter.next().unwrap();

        let w = WorkerThread::new(thread_id, local_assignment, &clusters_per_thread, messages_receive, messages_send);
        let shr = Arc::clone(&shared_state);

        // Note: use Arc on the SharedState to make the borrow checker happy
        workers.push(
            thread::spawn(||
                w.start(shr)));
    }

    #[cfg(test)]
    {
        thread::sleep(std::time::Duration::from_secs(1));
        for w in workers.iter() {
            assert!(w.is_finished());
        }
    }

    let finished_workers = workers.into_iter()
                            // Wait for the thread to finish
                            .map(|w| w.join().unwrap())
                            .collect::<Vec<_>>();
    let shared_state = Arc::into_inner(shared_state).unwrap();

    Clusters::from(finished_workers, shared_state)
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
    my_neighbours: Neighbours,
}

impl<T> WorkerThread<T> {
    fn new(
        thread_id:           ThreadId,
        starting_assignment: RemoteAssignment<T>,
        clusters_per_thread: &[usize],
        inward:              Receiver<Message<T>>,
        outwards:            Vec<Sender<Message<T>>>)
        -> Self
    {
        WorkerThread {
            thread_id,
            inward,
            outwards,
            my_assignment: WorkerAssignment::from(thread_id, starting_assignment, clusters_per_thread),
            my_neighbours: Neighbours::new(thread_id, clusters_per_thread)
        }
    }

    fn send_remote_assignments(&mut self) {
        for (dest, data) in self.my_assignment
                                .extract_remote_assignments()
                                .collect::<Vec<_>>()
                                .into_iter()
        {
            let message = Ok((self.thread_id, data));
            self.send_message(dest, message);
        }
        self.notify_others_done_with_assign();
    }

    fn notify_others_done_with_assign(&self) {
        for (_channel_id, channel) in self.outwards
                            .iter()
                            .enumerate()
                            .filter(|(other_id, _)| *other_id != self.thread_id)
        {
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
                    self.my_assignment.integrate_remote(remote_assignment);
                    received_any_changes = true;
                },
                Ok(Err(thread_eof)) => {
                    let was_there = not_received.remove(&thread_eof);
                    assert!(was_there, "Thread {} received EOF from thread {} twice", self.thread_id, thread_eof);
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

    fn num_clusters_assigned_to_me(&self) -> usize {
        self.my_assignment
            .num_local_clusters()
    }
}

impl<T: Point> WorkerThread<T> {
    fn start(mut self, shared_state: Arc<SharedState<T>>) -> Self {
        println!("T({}): start", self.thread_id);
        let mut i_have_changes = true;
        while shared_state.consensus_continue(i_have_changes) {
            // Note: this is local only, not read by anyone else
            println!("T({}): updating neighbours...", self.thread_id);
            self.update_centroid_neighbours(&shared_state.centroid_reader());
            println!("T({}): assigning points...", self.thread_id);
            let local_changes = self.assign_points_to_clusters(&shared_state);
            println!("T({}): sending remote assignments...", self.thread_id);
            self.send_remote_assignments();
            // This effectively acts like a sync barrier
            println!("T({}): receiving remote assignments...", self.thread_id);
            let remote_changes = self.receive_remote_assignments(shared_state.nthreads);
            i_have_changes = local_changes || remote_changes;
            println!("T({}): computing centroids...", self.thread_id);
            self.compute_centroids(&shared_state);
            // Note: the while-loop condition performs sync after this step
        }

        return self;
    }

    fn assign_points_to_clusters(&mut self, shared_state: &SharedState<T>) -> bool {
        assign_points_seq(&mut self.my_assignment, &self.my_neighbours, &shared_state.centroid_reader())
    }

    fn compute_centroids(&self, shared_state: &SharedState<T>) {
        // We only consider the points assigned to this thread
        let local_asg = self.my_assignment.get_local();
        let mut centroid_writer = shared_state.centroids
                                    .get_write_access(self.thread_id);

        for (i, points_in_cluster) in local_asg.iter_clusters() {
            centroid_writer[i].update_center(points_in_cluster);
        }
    }

    // This is local only
    // Note: we're not updating the centroids in this phase, only their neighbours, which are always local.
    fn update_centroid_neighbours(&mut self, centroid_reader: &CentroidReader<T>) {
        self.my_neighbours.update(self.thread_id, centroid_reader);
    }
}

/***************/
/* SharedState */
/***************/

struct SharedState<T> {
    nthreads:           usize,
    anyone_has_changes: AtomicBool,
    barrier:            Barrier,
    centroids:          RwCentroids<T>,
}

impl<T> SharedState<T> {
    fn new(nthreads: usize, centroids: Vec<Vec<Centroid<T>>>) -> Self {
        SharedState {
            nthreads,
            // WARNING: this starting value is very important
            // See consensus_continue()
            anyone_has_changes: AtomicBool::new(false),
            barrier:            Barrier::new(nthreads),
            centroids:          RwCentroids::new(centroids)
        }
    }

    // If any thread has changes, this returns true
    fn consensus_continue(&self, i_have_changes: bool) -> bool {
        println!("consensus_continue: calling thread {} changes", if i_have_changes { "has" } else { "does not have" });
        if i_have_changes {
            self.anyone_has_changes.store(true, ATOMIC_ORDERING);
        }

        // Wait for every thread to get to this point
        self.barrier.wait();

        let should_continue = self.anyone_has_changes.load(ATOMIC_ORDERING);

        // Make sure that every thread has read the value
        self.barrier.wait();

        if should_continue {
            // Reset the shared variable
            // Note: every thread does it when only one would need to.
            // Though technically if only one thread updates it, other thread could
            // get to call consensus_continue() before it's done writing
            self.anyone_has_changes.store(false, ATOMIC_ORDERING)
        }

        return should_continue;
    }

    fn centroid_reader(&self) -> CentroidReader<'_, T> {
        self.centroids.get_read_access()
    }
}

/********************/
/* Point assignment */
/********************/

#[derive(PartialEq, Eq, Clone, Copy)]
struct FullClusterId {
    thread_id:         ThreadId,
    cluster_in_thread: PartialClusterId
}
type FullCentroidId = FullClusterId;

impl FullClusterId {
    fn from(thread_id: ThreadId, cluster_in_thread: PartialClusterId) -> Self {
        FullClusterId {
            thread_id,
            cluster_in_thread
        }
    }
}

type PartialClusterId  = usize;
type PartialCentroidId = PartialClusterId;

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

    fn from(this_thread: ThreadId, local_assignment: RemoteAssignment<T>, clusters_per_thread: &[usize]) -> Self {
        let mut wasg = Self::new(this_thread, clusters_per_thread);
        // TODO use integrate_remote if possible
        *wasg.get_local_mut() = local_assignment;
        return wasg;
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

    fn integrate_remote(&mut self, remote_assignment: RemoteAssignment<T>) {
        self.get_local_mut()
            .merge(remote_assignment)
    }

    fn is_empty(&self) -> bool {
        self.thread_assignments
            .iter()
            .all(|asg| asg.is_empty())
    }

    fn assign(&mut self, cluster_id: FullCentroidId, x: T) {
        self.thread_assignments[cluster_id.thread_id]
            .assign(cluster_id.cluster_in_thread, x)
    }

    fn num_local_clusters(&self) -> usize {
        self.get_local()
            .num_clusters()
    }

    fn get_local(&self) -> &RemoteAssignment<T> {
        &self.thread_assignments[self.current_thread]
    }

    fn get_local_mut(&mut self) -> &mut RemoteAssignment<T> {
        &mut self.thread_assignments[self.current_thread]
    }
}

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

    fn merge(&mut self, mut other: Self) {
        assert_eq!(self.num_clusters(), other.num_clusters());
        self.0.iter_mut()
            .zip(other.0.iter_mut())
            .for_each(|(v, w)| v.append(w))
    }

    fn assign(&mut self, target_cluster: PartialCentroidId, x: T) {
        self.0[target_cluster].push(x)
    }

    fn iter_clusters(&self) -> impl Iterator<Item=(PartialClusterId, &Vec<T>)> {
        self.0.iter()
            .enumerate()
    }

    fn len(&self) -> usize {
        self.iter_clusters()
            .map(|(_, data)| data.len())
            .sum()
    }

    fn into_iter_clusters(self) -> impl Iterator<Item=(PartialCentroidId, Vec<T>)> {
        self.0.into_iter()
            .enumerate()
    }

    fn num_clusters(&self) -> usize {
        self.0.len()
    }
}

/* point assignment */

// Returns true if some points changed assignment, false otherwise
// TODO share with the sequential code
fn assign_points_seq<T: Point>(curr_asg: &mut WorkerAssignment<T>, all_neighbours: &Neighbours, centroid_reader: &CentroidReader<T>) -> bool {
    // 1. Extract the local points
    //    This must be the only non-empty section of the assignment
    let (this_thread, old_assignment) = curr_asg.extract_local_assignment();
    assert!(curr_asg.is_empty());
    let new_assignment = curr_asg;

    // 2. Move the points accordingly
    let mut some_change = false;
    let mut obvious_stay_count = 0;
    let mut neighbour_cutoff_count = 0;
    let mut moved_count = 0;
    let mut stayed_count = 0;
    for (partial_cid, x) in old_assignment.into_iter_clusters()
                            .flat_map(|(partial_cid, v)| v.into_iter()
                                                            .map(move |x| (partial_cid, x)))
    {
        // Note: we know for a fact that each data point starts assigned to the current thread
        let prev_cluster_id = FullCentroidId::from(this_thread, partial_cid);

        // Start with the current centroid
        let current_cluster = &centroid_reader[prev_cluster_id];
        let mut min_dist = current_cluster.0.dist(&x);
        let mut closest_idx = Some(prev_cluster_id);

        // Check if we can early stop
        // TODO remove this 'if', it should be duplicate of the c_to_c_dist test below
        if min_dist <= certainty_radius(partial_cid, all_neighbours) {
            obvious_stay_count += 1;
        }
        else {
            // TODO: explain this
            let cluster_cutoff = 2.0 * min_dist;

            // Go over the neighbouring centroids in distance order
            for other_cluster in all_neighbours.neighbours_of(partial_cid) {
                if other_cluster.neigh_dist > cluster_cutoff {
                    neighbour_cutoff_count += 1;
                    break;
                }

                let t_centroid = &centroid_reader[other_cluster.neigh_id];
                let t_dist = t_centroid.0.dist(&x);

                if t_dist < min_dist {
                    min_dist = t_dist;
                    closest_idx = Some(other_cluster.neigh_id);
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

        let new_cluster_id = closest_idx.unwrap();
        new_assignment.assign(new_cluster_id, x);

        if prev_cluster_id != new_cluster_id {
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

/*****************/
/* Centroid sync */
/*****************/

/* RwCentroids */
// First level is indexed by thread id
// Second Vec level is indexed by PartialCentroidId
struct RwCentroids<T> (Vec<RwLock<Vec<Centroid<T>>>>);

impl<T> RwCentroids<T> {
    fn new(centroids: Vec<Vec<Centroid<T>>>) -> Self {
        RwCentroids (
            centroids.into_iter()
                    .map(|thread_data| RwLock::new(thread_data))
                    .collect()
        )
    }

    fn get_read_access(&self) -> CentroidReader<'_, T> {
        let all_read_guards = self.0
                                .iter()
                                // Note: if one thread asks for read access, then all threads must be guaranteed
                                // to be in a phase where they only read. Hence we don't need to lock
                                .map(|lock| lock.try_read().unwrap())
                                .collect::<Vec<_>>();
        CentroidReader(all_read_guards)
    }

    fn get_write_access(&self, thread_id: ThreadId) -> CentroidWriter<'_, T> {
        // By the same argument as above, we don't really need to lock for write accesses
        self.0[thread_id]
            .try_write()
            .unwrap()
    }

    fn into_inner(self) -> Vec<Vec<Centroid<T>>> {
        self.0
            .into_iter()
            .map(|lock| lock.into_inner().unwrap())
            .collect()
    }
}

/* CentroidReader */
struct CentroidReader<'a, T> (Vec<RwLockReadGuard<'a, Vec<Centroid<T>>>>);

impl<'a, T> CentroidReader<'a, T> {
    fn get(&self, cluster_id: FullCentroidId) -> Option<&Centroid<T>> {
        self.0
            .get(cluster_id.thread_id)
            .and_then(|rv| rv.get(cluster_id.cluster_in_thread))
    }
}

impl<'a, T> Index<FullClusterId> for CentroidReader<'a, T> {
    type Output = Centroid<T>;

    fn index(&self, cluster_id: FullClusterId) -> &Self::Output {
        self.get(cluster_id).unwrap()
    }
}

/* CentroidWriter */
type CentroidWriter<'a, T> = RwLockWriteGuard<'a, Vec<Centroid<T>>>;

/************/
/* Centroid */
/************/

// Just a marker struct for readability
struct Centroid<T>(T);

impl<T: Point> Centroid<T> {
    // This is basically seq::compute_centroids()
    fn from(points: &[T]) -> Self {
        Centroid (
            T::mean(points)
        )
    }

    // This is basically seq::compute_centroids()
    fn update_center(&mut self, points: &[T]) {
        self.0 = T::mean(points);
    }
}

/* Neighbours */
// The first level indexes by PartialCentroidId
//   These must be local to the current thread
// The second level contains the ordered list of neighbours for the given local neighbour
//   Note: the neighbour might be remote
//   This level is ordered by the distance from the starting centroid, which is the second tuple item
struct Neighbours (Vec<Vec<NeighInfo>>);

struct NeighInfo {
    neigh_id:   FullCentroidId,
    neigh_dist: f64
}

impl Neighbours {
    // Adapted from seq::init_neighbours
    fn new(this_thread: ThreadId, clusters_per_thread: &[usize]) -> Self {
        let num_local_clusters = clusters_per_thread[this_thread];
        let nthreads = clusters_per_thread.len();

        let mut all_neighbours = Vec::with_capacity(num_local_clusters);

        for src_partial_id in 0..num_local_clusters {
            let src_id = FullCentroidId::from(this_thread, src_partial_id);
            let neighbour_ids = (0..nthreads)
                                    .flat_map(|dst_thread| (0..clusters_per_thread[dst_thread])
                                                            .map(move |dst_cluster_in_thread| FullCentroidId::from(dst_thread, dst_cluster_in_thread)))
                                    .filter(|dst_id| dst_id != &src_id);
            let neigh_list = neighbour_ids.map(|dst_id| NeighInfo { neigh_id: dst_id, neigh_dist: f64::INFINITY});
            all_neighbours.push(neigh_list.collect());
        }

        Neighbours(all_neighbours)
    }

    // Adapted from seq::compute_neighbours
    fn update<T: Point>(&mut self, this_thread: ThreadId, centroid_reader: &CentroidReader<T>) {
        let mut sort_count = 0;
        // Go over all the local centroid neighbour lists
        for (partial_cid, neigh_list) in self.0.iter_mut().enumerate() {
            let local_cid = FullCentroidId::from(this_thread, partial_cid);
            let local_centroid = &centroid_reader[local_cid];

            // Update the distance to each neighbour for this local centroid
            for neigh_info in neigh_list.iter_mut() {
                let neigh_centroid = &centroid_reader[neigh_info.neigh_id];
                neigh_info.neigh_dist = local_centroid.0.dist(&neigh_centroid.0);
                assert!(neigh_info.neigh_dist.is_finite());
            }

            // Optimization: consider it "good enough" if the first sqrt(nclusters) are sorted
            // This leads to imprecision. Though when we get to the point where the first sqrt(nclusters)
            // remain sorted, we can assume that the points don't go so far in the list of neighbours anymore
            let check_n_sorted = (neigh_list.len() as f64).sqrt() as usize;
            let already_sorted = neigh_list
                                    .windows(2)
                                    .take(check_n_sorted)
                                    .find(|xs| xs[0].neigh_dist > xs[1].neigh_dist)
                                    .is_none();

            if !already_sorted {
                sort_count += 1;
                // Optimization: use unstable sorting
                // We don't care about the stability of the ordering, we're interested in speed
                neigh_list.sort_unstable_by(|a, b| a.neigh_dist.partial_cmp(&b.neigh_dist)
                                                                .ok_or_else(|| format!("{} <> {}", a.neigh_dist, b.neigh_dist))
                                                                .unwrap());
            }
        }
        println!("Sorted {} times", sort_count);
    }

    fn neighbours_of(&self, local_cluster_id: PartialCentroidId) -> &[NeighInfo] {
        &self.0[local_cluster_id]
    }
}

// Adapted from seq::Clusters::certainty_radius
fn certainty_radius(local_cid: PartialCentroidId, all_neighbours: &Neighbours) -> f64 {
    all_neighbours.0[local_cid]
        .get(0)
        .map(|first_neigh| first_neigh.neigh_dist / 2.0)
        .unwrap_or(f64::INFINITY)  // if a node has no neighbours, it has an infinite certainty radius
}

//type Neighbours = Vec<Vec<(usize, f64)>>;

/*fn init_neighbours<T: Point>(centroids: &[T]) -> Neighbours {
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
}*/

/*fn update_neighbours<T: Point>(clusters: &mut Clusters<T>) {
    // Optimization: don't clear the buffer and keep it in the same order
    compute_neighbours(&clusters.centroids, &mut clusters.neighbours);
}*/

/******************/
/* Initialization */
/******************/


// TODO change this for a simpler vector chunk approach
fn init_assignments<T, I>(points: &mut I, clusters_per_thread: &[usize]) -> Vec<RemoteAssignment<T>>
    where I: Iterator<Item = T>,
        T: Point
{
    let nthreads = clusters_per_thread.len();

    let mut init_assignment: Vec<RemoteAssignment<T>> = (0..nthreads).map(|i| RemoteAssignment::new(clusters_per_thread[i])).collect();

    let mut assignee = (0..nthreads)
                        .flat_map(|i| (0..clusters_per_thread[i])
                                        .map(move |j| (i, j)))
                        .cycle();

    for x in points {
        let (i, j) = assignee.next().unwrap();
        init_assignment[i].assign(j, x);
    }

    return init_assignment;
}

// Initialize the centroids for a single thread
// TODO share this with the sequential code
fn init_centroids<T: Point>(assignment: &RemoteAssignment<T>) -> Vec<Centroid<T>> {
    let nclusters = assignment.len();

    assignment.iter_clusters()
        .map(|(_, cluster_data)| Centroid::from(cluster_data))
        .collect()
}

/* centroids */

/*type Assignment<T> = Vec<Vec<T>>;

fn update_centroids<T: Point>(clusters: &mut Clusters<T>) {
    // Note: Vec::clear() keeps the underlying allocated buffer, which is what we want
    clusters.centroids.clear();
    compute_centroids(&clusters.assignment, &mut clusters.centroids);
}*/

/***************************/
/* Public API for Clusters */
/***************************/

pub struct Clusters<T> {
    clustered_data: Vec<RemoteAssignment<T>>,
    centroids:      Vec<Vec<Centroid<T>>>,
}

impl<T> Clusters<T> {
    fn from(finished_workers: Vec<WorkerThread<T>>, shared_state: SharedState<T>) -> Self {
        Clusters {
            clustered_data: finished_workers.into_iter()
                                .map(|mut w| w.my_assignment.extract_local_assignment().1)
                                .collect(),
            centroids: shared_state.centroids.into_inner(),
        }
    }

    fn iter(&self) -> impl Iterator<Item=(&T, &Vec<T>)> {
        self.centroids
            .iter()
            .zip(self.clustered_data.iter())
            .flat_map(|(centroids_in_thread, asgs_in_thread)|
                centroids_in_thread.iter()
                    .map(|c| &c.0)
                    .zip(asgs_in_thread.iter_clusters()
                            .map(|(_id, data)| data)))
    }

    fn num_centroids(&self) -> usize {
        self.centroids
            .iter()
            .map(|v| v.len())
            .sum()
    }

    fn centroids(&self) -> impl Iterator<Item=&T> {
        self.centroids
            .iter()
            .flat_map(|v| v.iter()
                            .map(|c| &c.0))
    }

    fn clustered_data(&self) -> impl Iterator<Item=&Vec<T>> {
        self.clustered_data
            .iter()
            .flat_map(|asgs_in_thread|
                asgs_in_thread.iter_clusters()
                            .map(|(_id, data)| data))
    }
}


/**************/
/* Unit tests */
/**************/

#[cfg(test)]
mod test {
    use super::*;
    use super::super::test::square_centered_at;
    use std::collections::HashSet;

    fn assert_centroids<T>(clusters: &Clusters<T>, expected_centroids: &[T])
        where T: Clone + Eq + std::hash::Hash + std::fmt::Debug
    {
        assert_eq!(clusters.num_centroids(), expected_centroids.len());

        let centroids = HashSet::<_>::from_iter(clusters.centroids().cloned());
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

        for (centroid, data_in_cluster) in clusters.iter() {
            assert_eq!(data_in_cluster, std::slice::from_ref(centroid));
        }
    }

    #[test]
    fn square1() {
        let data = square_centered_at((0, 0));
        let ndata = data.len();
        println!("{:?}", &data);
        let clusters = super::cluster(&mut data.into_iter(), 1);
        assert_centroids(&clusters, &[(0, 0)][..]);

        let mut it = clusters.clustered_data();
        assert_eq!(it.next().map(|asg| asg.len()), Some(ndata));
        assert_eq!(it.next(), None);
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
        assert_centroids(&clusters, &sq_centers[..]);
    }

    /*#[test]
    fn radii() {
        let data = [(0, 0), (1, 0)];
        let clusters = super::cluster(&mut data.into_iter(), data.len());
        // Note: the certainty radius is half the distance to the closest centroid
        assert_eq!(clusters.certainty_radius(0), 0.5);
        assert_eq!(clusters.certainty_radius(1), 0.5);
    }*/
}
