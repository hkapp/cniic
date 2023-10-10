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
