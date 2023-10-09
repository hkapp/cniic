use std::cmp::Ordering;
use std::collections::{HashMap, BinaryHeap};
use std::hash::Hash;
use std::ops::Add;

use crate::bit::Bit;
use crate::bit;

/* Enc */

struct Code {
    bit_count:   u16,
    packed_bits: Vec<u8>,
}

impl From<&[Bit]> for Code {
    fn from(bits: &[Bit]) -> Self {
        let mut packed_bits = Vec::new();
        let mut rem_bits = bits.len();
        let mut pos = 0;

        // Pack the bits 8-by-8 as long as possible
        while rem_bits >= 8 {
            let full_byte = bit::byte_from_slice(&bits[pos..(pos+8)]).unwrap();
            packed_bits.push(full_byte);
            pos += 8;
            rem_bits -= 8;
        }

        // Pack the remaining bits
        if rem_bits > 0 {
            let mut incomplete_byte = 0u8;
            while pos < bits.len() {
                incomplete_byte <<= 1;
                incomplete_byte &= bits[pos].byte();
                pos += 1;
            }
        }

        Code {
            bit_count: bits.len().try_into().unwrap(),
            packed_bits
        }
    }
}

struct Enc<T> {
    codes: HashMap<T, Code>
}

impl<T: Hash + Eq + Clone> From<&Dec<T>> for Enc<T> {
    fn from(enc: &Dec<T>) -> Self {
        let mut codes = HashMap::new();
        for (bits, symbol) in enc.trie.iter() {
            codes.insert(symbol.clone(), Code::from(bits));
        }
        Enc {
            codes
        }
    }
}

/* Dec */

struct Dec<T> {
    trie: BinTrie<T>
}

/* build */

fn build<I, T, N>(freq_items: I) -> (Enc<T>, Dec<T>)
    where I: Iterator<Item = (T, N)>,
        T: Eq + Hash + Clone,
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

/* BinTrie */

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

    // Iterate over all the elements and representations of this trie
    fn iter(&self) -> BinTrieIter<T> {

    }
}

struct BinTrieIter<'a, T> {
    stack:    Vec<&'a BinTrie<T>>,
    curr_rep: Vec<Bit>
}

impl<'a, T> Iterator for BinTrieIter<'a, T> {
    type Item = (&'a [Bit], &'a T);

    fn next(&mut self) -> Option<Self::Item> {

    }
}
