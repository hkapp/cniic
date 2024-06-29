use std::collections::HashMap;
use std::hash::Hash;

pub fn count_freqs<I, T>(items: I) -> HashMap<T, u64>
    where I: Iterator<Item = T>,
        T: Eq + Hash
{
    let mut freqs = HashMap::new();
    for x in items {
        freqs
            .entry(x)
            .and_modify(|n| *n += 1)
            .or_insert(1);
    }
    return freqs;
}

pub fn default_array<const N: usize, T: Default>() -> [T; N] {
    // From https://www.reddit.com/r/rust/comments/mg1crv/comment/gssaazc/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
    // let mut array: [MaybeUninit<T>; N] = unsafe {
    //     MaybeUninit::uninit().assume_init()
    // };
    // for elem in array.iter_mut() {
    //     *elem = MaybeUninit::new(T::default());
    // }
    // unsafe {
    //     // std::mem::transmute::<_, [T; N]>(array)
    //     *(&array as *const [MaybeUninit<T>; N] as *const [T; N])
    // }

    (0..N)
        .map(|_| T::default())
        .collect::<Vec<T>>()
        .try_into()
        .map_err(|v: Vec<T>| v.len())
        .unwrap()
}
