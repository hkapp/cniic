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
            codes.insert(symbol.clone(), Code::from(&bits as &[Bit]));
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
        BinTrieIter::new(self)
    }
}

struct BinTrieIter<'a, T> {
    stack:    Vec<&'a BinTrie<T>>,
    curr_rep: Vec<Bit>
}

impl<'a, T> Iterator for BinTrieIter<'a, T> {
    type Item = (Vec<Bit>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_over() {
            return None;
        }

        // 1. Save the next iterator value
        let rep = self.curr_rep.clone();
        let sym =
            match self.stack.last() {
                Some(BinTrie::Leaf(t)) => t,
                _                      => panic!("Logic error: no value in iterator"),
            };

        // 2. Move to the next leaf:
        //    Pop parents we're to the right of
        //    Move right once
        //    Move to the leaf left-most in this new sub-tree
        self.pop_parents();
        if !self.is_over() {
            self.right_once();
            self.all_the_way_left();
        }

        return Some((rep, sym));
    }
}

impl<'a, T> BinTrieIter<'a, T> {
    fn new(trie: &'a BinTrie<T>) -> Self {
        let mut bti =
            BinTrieIter {
                stack:    vec![trie],
                curr_rep: Vec::new()
            };
        bti.all_the_way_left();
        return bti;
    }

    fn is_over(&self) -> bool {
        self.stack.len() == 0
    }

    fn all_the_way_left(&mut self) {
        let mut curr_node: &'a BinTrie<T> = self.stack.last().unwrap();
        while let BinTrie::Branch(left, _) = curr_node {
            self.stack.push(left);
            self.curr_rep.push(Bit::Zero);
            curr_node = left;
        }
    }

    fn right_once(&mut self) {
        match self.stack.last().unwrap() {
            BinTrie::Branch(_, right) => {
                self.stack.push(right);
                self.curr_rep.push(Bit::One);
            }
            BinTrie::Leaf(_) => {
                // This is actually guaranteed by the code logic
                panic!("We must have a right child")
            }
        }
    }

    fn pop_parents(&mut self) {
        // Pop parents we're to the right of
        while self.curr_rep.last() == Some(&Bit::One) {
            let _  = self.stack.pop();
            let _  = self.curr_rep.pop();
        }
    }
}
