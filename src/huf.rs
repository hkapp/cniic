use std::cmp::Ordering;
use std::collections::{HashMap, BinaryHeap};
use std::hash::Hash;
use std::ops::Add;

use crate::bit::{Bit, BitOrder};
use crate::bit;

pub const BIT_ORDER: BitOrder = BitOrder::MsbFirst;

/* Enc */

// The bits in this data structure are packed into bytes
// This packing is done MSB first
// TODO consider moving this to the bit module (e.g. BitArray)
// Note: then it can be used for the IoBitWriter
#[derive(PartialEq, Eq, Debug)]
struct Code {
    full_bytes:      Vec<u8>,
    partial_count:   u8,
    partial_byte:    u8,
}

impl From<&[Bit]> for Code {
    fn from(bits: &[Bit]) -> Self {
        let mut packed_bits = Vec::new();
        let mut rem_bits = bits.len();
        let mut pos = 0;

        // Pack the bits 8-by-8 as long as possible
        /* TODO make this into a bit module interface */
        while rem_bits >= 8 {
            let full_byte = bit::byte_from_slice(&bits[pos..(pos+8)], BIT_ORDER).unwrap();
            packed_bits.push(full_byte);
            pos += 8;
            rem_bits -= 8;
        }

        // Pack the remaining bits
        let mut incomplete_byte = 0u8;
        while pos < bits.len() {
            bit::push_bit(&mut incomplete_byte, bits[pos], BIT_ORDER);
            pos += 1;
        }

        Code {
            full_bytes:    packed_bits,
            partial_count: rem_bits as u8,
            partial_byte:  incomplete_byte
        }
    }
}

pub struct Enc<T> {
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

impl<T: Hash + Eq> Enc<T> {
    pub fn encode<W: bit::WriteBit>(&self, symbol: &T, writer: &mut W) -> Option<()> {
        let code = self.codes.get(symbol);
        if code.is_none() {
            return None;
        }
        let code = code.unwrap();
        if code.full_bytes.len() == 0 && code.partial_count == 0 {
            return Some(());
        }

        for byte in code.full_bytes.iter() {
            writer.write_byte(*byte)
                .map_err(|e| eprintln!("{:?}", e))
                .ok()?;
        }

        /* Read the remaining bits:
         *   xxxx x010
         *         012
         */
        let starting_bit: u8 = 8 - code.partial_count;
        for j in starting_bit..8 {
            writer.write(bit::nth(code.partial_byte, j, BIT_ORDER))
                .map_err(|e| eprintln!("{:?}", e))
                .ok()?;
        }

        return Some(());
    }
}

/* Dec */

#[derive(PartialEq, Eq, Debug)]
pub struct Dec<T> {
    trie: BinTrie<T>
}

impl<T> Dec<T> {
    pub fn decode<I>(&self, input: &mut I) -> Option<&T>
        where I: Iterator<Item = Bit>
    {
        self.trie.lookup(input)
    }
}

/* build */

pub fn build<I, T, N>(freq_items: I) -> (Enc<T>, Dec<T>)
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
            // Suffix has a reverse ordering to have a min_heap later
            self.freq.partial_cmp(&other.freq).map(Ordering::reverse)
        }
    }
    impl<T, N: Ord> Ord for Suffix<T, N> {
        fn cmp(&self, other: &Self) -> Ordering {
            // Suffix has a reverse ordering to have a min_heap later
            self.freq.cmp(&other.freq).reverse()
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

#[derive(PartialEq, Eq, Debug)]
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

    fn lookup<I>(&self, input: &mut I) -> Option<&T>
        where I: Iterator<Item = Bit>
    {
        let mut curr_node = self;
        loop {
            match curr_node {
                BinTrie::Branch(left, right) => {
                    curr_node =
                        match input.next() {
                            Some(Bit::Zero) => left,
                            Some(Bit::One)  => right,
                            None            => return None, /* EOF */
                        };
                }
                BinTrie::Leaf(value) => {
                    return Some(value);
                }
            }
        }
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
        // Pop at least one parent
        let _ = self.stack.pop();
        let mut prev_dir = self.curr_rep.pop();

        // Pop all the parents we were to the right of
        while prev_dir == Some(Bit::One) {
            let _ = self.stack.pop();
            prev_dir = self.curr_rep.pop();
        }
    }
}

/* Serialize / Deserialize */
use std::io;
use crate::ser::{Serialize, Deserialize};

const SER_ENUM_LEAF:   u8 = 0;
const SER_ENUM_BRANCH: u8 = 1;

impl<T: Serialize> Serialize for Dec<T> {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        self.trie.serialize(writer)
    }
}

impl<T: Serialize> Serialize for BinTrie<T> {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        match self {
            BinTrie::Leaf(v) => {
                SER_ENUM_LEAF.serialize(writer)?;
                v.serialize(writer)?;
                Ok(())
            }
            BinTrie::Branch(left, right) => {
                SER_ENUM_BRANCH.serialize(writer)?;
                left.serialize(writer)?;
                right.serialize(writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Deserialize> Deserialize for Dec<T> {
    fn deserialize<I: Iterator<Item = u8>>(stream: &mut I) -> Option<Self> {
        Deserialize::deserialize(stream)
            .map(|trie| Dec { trie })
    }
}

impl<T: Deserialize> Deserialize for BinTrie<T> {
    fn deserialize<I: Iterator<Item = u8>>(stream: &mut I) -> Option<Self> {
        let decider: u8 = Deserialize::deserialize(stream)?;
        match decider {
            SER_ENUM_LEAF => {
                let value = Deserialize::deserialize(stream)?;
                Some(BinTrie::Leaf(value))
            }
            SER_ENUM_BRANCH => {
                let left = Deserialize::deserialize(stream)?;
                let right = Deserialize::deserialize(stream)?;
                Some(BinTrie::Branch(Box::new(left), Box::new(right)))
            }
            _ => {
                None // Failed to deserialize
            }
        }
    }
}

/* Unit tests */

#[cfg(test)]
mod tests {
    use super::bit;
    use bit::{Bit, WriteBit};
    use crate::ser::{Serialize, Deserialize};
    use super::*;

    fn huf_abc() -> (super::Enc<char>, super::Dec<char>) {
        super::build([('a', 2), ('b', 1), ('c', 1)].into_iter())
    }

    #[test]
    fn builder_is_sane() {
        let (enc, _) = huf_abc();
        assert_eq!(enc.codes.len(), 3);
        assert_eq!(enc.codes.len(), 3);
    }

    #[test]
    fn bintrie_iter1() {
        let trie = super::BinTrie::leaf(1);
        let mut bti = trie.iter();
        assert!(bti.next().is_some());
        assert!(bti.next().is_none());
    }

    #[test]
    fn bintrie_iter2() {
        let trie =
            super::BinTrie::compose(
                Box::new(super::BinTrie::leaf(0)),
                Box::new(super::BinTrie::leaf(1)));
        let mut bti = trie.iter();

        assert_eq!(bti.next(), Some((vec![Bit::Zero], &0)));
        assert_eq!(bti.next(), Some((vec![Bit::One], &1)));
        assert!(bti.next().is_none());
    }

    #[test]
    fn code_lens1() {
        let (enc, _) = huf_abc();
        /* FIXME use BitArray::bit_count() when available */
        assert_eq!(enc.codes.get(&'a').map(|c| c.partial_count), Some(1));
        assert_eq!(enc.codes.get(&'b').map(|c| c.partial_count), Some(2));
        assert_eq!(enc.codes.get(&'c').map(|c| c.partial_count), Some(2));
    }

    fn enc_str(enc: &Enc<char>, s: &str) -> Vec<u8> {
        let mut bw = bit::IoBitWriter::new(Vec::new(), super::BIT_ORDER);
        for c in s.chars() {
            enc.encode(&c, &mut bw);
        }
        assert!(bw.pad_and_flush().is_ok());
        return bw.into_inner();
    }

    #[test]
    fn enc_dec1() {
        let (enc, dec) = huf_abc();
        let input = "a";

        let message = enc_str(&enc, input);
        let byte_iter = message.into_iter();
        let mut bit_iter = byte_iter.flat_map(
                            |n| bit::bit_array(n, super::BIT_ORDER).into_iter());

        for c in input.chars() {
            let decoded = dec.decode(&mut bit_iter);
            assert_eq!(decoded, Some(&c));
        }
    }

    #[test]
    fn enc_dec2() {
        let (enc, dec) = huf_abc();
        let input = "abcabcaabbcc";

        let message = enc_str(&enc, input);
        let byte_iter = message.into_iter();
        let mut bit_iter = byte_iter.flat_map(
                            |n| bit::bit_array(n, super::BIT_ORDER).into_iter());

        for c in input.chars() {
            let decoded = dec.decode(&mut bit_iter);
            assert_eq!(decoded, Some(&c));
        }
    }

    #[test]
    fn enc_dec3() {
        let (enc, dec) = huf_abc();
        let input = "abcabcaabbcc";

        let mut ser_dec = Vec::new();
        assert!(dec.serialize(&mut ser_dec).is_ok());
        let dec2 = Deserialize::deserialize(&mut ser_dec.into_iter());
        assert!(dec2.is_some());
        let dec2: super::Dec<char> = dec2.unwrap();

        let message = enc_str(&enc, input);
        let byte_iter = message.into_iter();
        let mut bit_iter = byte_iter.flat_map(
                            |n| bit::bit_array(n, super::BIT_ORDER).into_iter());

        for c in input.chars() {
            let decoded = dec2.decode(&mut bit_iter);
            assert_eq!(decoded, Some(&c));
        }
    }

    #[test]
    fn ser1() {
        let (_enc, dec) = huf_abc();

        let mut vec = Vec::new();
        assert!(dec.serialize(&mut vec).is_ok());

        let dec2 = Deserialize::deserialize(&mut vec.into_iter());
        assert_eq!(dec2, Some(dec));
    }

    #[test]
    fn bit_packing0() {
        let packed = Code::from(&[Bit::Zero][..]);
        let expected = Code {
            full_bytes:    Vec::new(),
            partial_count: 1,
            partial_byte:  0
        };
        assert_eq!(packed, expected);
    }

    #[test]
    fn bit_packing1() {
        let packed = Code::from(&[Bit::One][..]);
        let expected = Code {
            full_bytes:    Vec::new(),
            partial_count: 1,
            partial_byte:  1
        };
        assert_eq!(packed, expected);
    }

    #[test]
    fn bit_packing2() {
        // 0xf0
        let bit_array = [
            Bit::One,  Bit::One,  Bit::One,  Bit::One,
            Bit::Zero, Bit::Zero, Bit::Zero, Bit::Zero,
        ];
        let packed = Code::from(&bit_array[..]);
        let expected = Code {
            full_bytes:    vec![0xf0],
            partial_count: 0,
            partial_byte:  0
        };
        assert_eq!(packed, expected);
    }

    #[test]
    fn bit_packing3() {
        // 0xf0 + 0b1
        let bit_array = [
            Bit::One,  Bit::One,  Bit::One,  Bit::One,
            Bit::Zero, Bit::Zero, Bit::Zero, Bit::Zero,
            Bit::One,
        ];
        let packed = Code::from(&bit_array[..]);
        let expected = Code {
            full_bytes:    vec![0xf0],
            partial_count: 1,
            partial_byte:  1
        };
        assert_eq!(packed, expected);
    }

    #[test]
    fn encode1() {
        let mut codes = HashMap::new();

        // 0b010
        let bit_array = [Bit::Zero,  Bit::One,  Bit::Zero];
        codes.insert('a', Code::from(&bit_array[..]));

        // 0xf0 + 0b011
        let bit_array = [
            Bit::One,  Bit::One,  Bit::One,  Bit::One,
            Bit::Zero, Bit::Zero, Bit::Zero, Bit::Zero,
            Bit::Zero, Bit::One,  Bit::One,
        ];
        codes.insert('b', Code::from(&bit_array[..]));

        // 0b00
        let bit_array = [Bit::Zero,  Bit::Zero];
        codes.insert('c', Code::from(&bit_array[..]));

        let enc = Enc { codes };
        let message = enc_str(&enc, "abc");
        assert_eq!(message, vec![0x5e, 0x0c])
    }

    #[test]
    fn encode2() {
        let mut codes = HashMap::new();

        // 0xf0
        let bit_array = [
            Bit::One,  Bit::One,  Bit::One,  Bit::One,
            Bit::Zero, Bit::Zero, Bit::Zero, Bit::Zero,
        ];
        codes.insert('a', Code::from(&bit_array[..]));

        let enc = Enc { codes };
        let message = enc_str(&enc, "a");
        assert_eq!(message, vec![0xf0])
    }
}
