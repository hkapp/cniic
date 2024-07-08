use std::cmp::Ordering;
use std::collections::{HashMap, BinaryHeap};
use std::hash::Hash;
use std::ops::Add;
use std::io;
use crate::ser::{Serialize, Deserialize};
use crate::bit::{self, bit_reader, Bit, BitArray, BitOrder, IoBitWriter, WriteBit};
use crate::utils;

/// Perform the entire Hufman encoding process.
///
/// This function will:
///  1. build the Hufman dictionary
///  2. serialize the decoder in the given `writer`
///  3. serialize all the values coming from `new_iter`
///
/// Because Hufman encoding takes place in two stages, the input data must
/// be read twice.
/// This is why `new_iter` is a function that returns an iterator instead
/// of an iterator.
/// The function `new_iter` will be called exactly twice.
pub fn encode_all<I, T, W, F>(new_iter: F, writer: &mut W) -> io::Result<()>
    where
        I: Iterator<Item = T>,
        T: Eq + Hash + Clone + Serialize,
        W: io::Write,
        F: Fn() -> I
{
    // 1. Build the Hufman encoder
    let item_freqs = utils::count_freqs(new_iter());
    let (enc, dec) = build(item_freqs.into_iter());

    // 2. Serialize the Hufman decoder
    dec.serialize(writer)?;

    // 3. Write the payload
    let mut bit_writer = IoBitWriter::new(writer, BIT_ORDER);
    for symbol in new_iter() {
        enc.encode(&symbol, &mut bit_writer).unwrap();
    }
    bit_writer.pad_and_flush()?;
    Ok(())
}

/// Decodes a stream of values previously serialized by [encode_all].
pub fn decode_all<I: Iterator<Item = u8>, T: Deserialize>(mut input: I) -> Option<DecStream<impl Iterator<Item = Bit>, T>> {
    // We expect the decoder to have been serialized just before the data
    let dec = <Dec<T>>::deserialize(&mut input)?;
    let bits = bit_reader(input, BIT_ORDER);

    let ds = DecStream::new(dec, bits);
    Some(ds)
}

const BIT_ORDER: BitOrder = BitOrder::MsbFirst;

/// Builds an encoder/decoder pair from item/frequency pairs
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

/* Enc */

struct Enc<T> {
    codes: HashMap<T, BitArray>
}

impl<T: Hash + Eq + Clone> From<&Dec<T>> for Enc<T> {
    fn from(enc: &Dec<T>) -> Self {
        let mut codes = HashMap::new();
        for (bits, symbol) in enc.trie.iter() {
            codes.insert(symbol.clone(), BitArray::from_slice(&bits as &[Bit], BIT_ORDER));
        }
        Enc {
            codes
        }
    }
}

impl<T: Hash + Eq> Enc<T> {
    fn encode<W: bit::WriteBit>(&self, symbol: &T, writer: &mut W) -> Option<()> {
        let code = self.codes.get(symbol)?;
        if code.len() == 0 {
            return Some(());
        }

        writer.write_arr(code)
            .map_err(|e| eprintln!("{:?}", e))
            .ok()
    }
}

/* Dec */

#[derive(PartialEq, Eq, Debug)]
struct Dec<T> {
    trie: BinTrie<T>
}

impl<T> Dec<T> {
    fn decode<I>(&self, input: &mut I) -> Option<&T>
        where I: Iterator<Item = Bit>
    {
        self.trie.lookup(input)
    }
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

/* DecStream */

pub struct DecStream<I, T> {
    dec:  Dec<T>,
    bits: I
}

impl<I, T> DecStream<I, T> {
    fn new(dec: Dec<T>, bits: I) -> Self {
        DecStream {
            dec,
            bits
        }
    }
}

impl<I: Iterator<Item = Bit>, T: Clone> Iterator for DecStream<I, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.dec
            .decode(&mut self.bits)
            .cloned()
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

        assert_eq!(enc.codes.get(&'a').map(|c| c.len()), Some(1));
        assert_eq!(enc.codes.get(&'b').map(|c| c.len()), Some(2));
        assert_eq!(enc.codes.get(&'c').map(|c| c.len()), Some(2));
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
    fn encode1() {
        let mut codes = HashMap::new();

        // 0b010
        let bit_array = [Bit::Zero,  Bit::One,  Bit::Zero];
        codes.insert('a', BitArray::from_slice(&bit_array[..], BIT_ORDER));

        // 0xf0 + 0b011
        let bit_array = [
            Bit::One,  Bit::One,  Bit::One,  Bit::One,
            Bit::Zero, Bit::Zero, Bit::Zero, Bit::Zero,
            Bit::Zero, Bit::One,  Bit::One,
        ];
        codes.insert('b', BitArray::from_slice(&bit_array[..], BIT_ORDER));

        // 0b00
        let bit_array = [Bit::Zero,  Bit::Zero];
        codes.insert('c', BitArray::from_slice(&bit_array[..], BIT_ORDER));

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
        codes.insert('a', BitArray::from_slice(&bit_array[..], BIT_ORDER));

        let enc = Enc { codes };
        let message = enc_str(&enc, "a");
        assert_eq!(message, vec![0xf0])
    }
}
