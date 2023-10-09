use std::cmp::Ordering;
use std::collections::{HashMap, BinaryHeap};
use std::hash::Hash;
use std::ops::Add;

type Code = u8;

struct Enc<T> {
    codes: HashMap<T, Code>
}

impl<T> From<&Dec<T>> for Enc<T> {
    fn from(enc: &Dec<T>) -> Self {

    }
}

struct Dec<T> {
    trie: BinTrie<T>
}

fn build<I, T, N>(freq_items: I) -> (Enc<T>, Dec<T>)
    where I: Iterator<Item = (T, N)>,
        T: Eq + Hash,
        N: Add<Output = N> + Ord
{
    struct Suffix<T, N> {
        freq: N,
        tree: BinTrie<T>
    }

    impl<T, N: PartialEq> PartialEq for Suffix<T, N> {
        fn eq(&self, other: &Self) -> bool {
            self.freq.eq(&other.freq)
        }
    }
    impl<T, N: Eq> Eq for Suffix<T, N> { }
    impl<T, N: PartialOrd> PartialOrd for Suffix<T, N> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.freq.partial_cmp(&other.freq)
        }
    }
    impl<T, N: Ord> Ord for Suffix<T, N> {
        fn cmp(&self, other: &Self) -> Ordering {
            self.freq.cmp(&other.freq)
        }
    }

    impl<T, N> From<(T, N)> for Suffix<T, N> {
        fn from(value: (T, N)) -> Self {
            Suffix {
                freq: value.1,
                tree: BinTrie::leaf(value.0)
            }
        }
    }

    let mut min_heap =
        BinaryHeap::from_iter(
            freq_items.map(Suffix::from));
    assert!(min_heap.len() > 0);
    while min_heap.len() > 1 {
        let left = min_heap.pop().unwrap();
        let right = min_heap.pop().unwrap();
        let new_tree = BinTrie::compose(Box::from(left.tree), Box::from(right.tree));
        let new_freq = left.freq + right.freq;
        let new_suff = Suffix {
            freq: new_freq,
            tree: new_tree
        };
        min_heap.push(new_suff);
    }
    let final_tree = min_heap.pop().unwrap().tree;
    let dec = Dec {
        trie: final_tree
    };
    let enc = Enc::from(&dec);
    (enc, dec)
}

enum BinTrie<T> {
    Leaf(T),
    Branch(Box<BinTrie<T>>, Box<BinTrie<T>>)
}

impl<T> BinTrie<T> {
    fn leaf(value: T) -> Self {
        BinTrie::Leaf(value)
    }

    fn compose(left: Box<Self>, right: Box<Self>) -> Self {
        BinTrie::Branch(left, right)
    }
}
