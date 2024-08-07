use std::{collections::HashMap, io, slice};
use bytesize::ByteSize;

use crate::{ser::{Deserialize, Serialize}, utils::default_array};

const ZIP_SPECIAL_EOF: Symbol = Symbol::MAX;

pub fn zip_dict_encode<I: Iterator<Item=u8>, W: io::Write>(bytes: I, output: &mut W) -> io::Result<()> {
    let mut zip = DictEncoder::new(bytes);

    while let Some((symbol1, symbol2)) = zip.next_pair() {
        symbol1.serialize(output)?;
        symbol2.serialize(output)?;
    }

    zip.trie.print_stats();

    Ok(())
}

pub fn zip_dict_decode<I: Iterator<Item=u8>>(encoded_stream: I) -> DictDecoder<I> {
    DictDecoder::new(encoded_stream)
}

/* DictEncoder */

struct DictEncoder<I> {
    input:   Input<I>,
    trie:    TrieMap<Symbol>,
    abbrev:  Abbrev
}

type Symbol = u16;

impl<I> DictEncoder<I> {
    fn new(input: I) -> Self {
        let mut zip = DictEncoder {
            input:   Input::new(input),
            trie:    TrieMap::new(),
            abbrev: Abbrev::new(),
        };

        // Important: use an inclusive range
        for b in 0x00..=0xff {
            zip.create_symbol(slice::from_ref(&b));
        }
        assert_eq!(zip.abbrev.counter, 0x0100);

        return zip;
    }

    fn create_symbol(&mut self, seq: &[u8]) {
        match self.abbrev.next() {
            Some(new_code) => {
                self.trie.insert(seq, new_code);
            }
            None => {
                // No more available symbols
                // Nothing to do
            }
        }
    }
}

impl<I: Iterator<Item=u8>> DictEncoder<I> {
    fn next_pair(&mut self) -> Option<(Symbol, Symbol)> {
        let (symbol1, seq1) = self.find_symbol();

        if symbol1.is_none() {
            // That can only happen if the input stream is empty
            // 1 byte always yields a symbol
            assert!(seq1.is_empty());
            assert!(self.input.left_over.is_empty() && self.input.input.next().is_none());
            return None;
        }
        let symbol1 = symbol1.unwrap();
        assert!(seq1.len() > 0);

        let (symbol2, mut seq2) = self.find_symbol();

        if symbol2.is_none() {
            // Here too, this can only happen if the input stream is empty
            assert!(seq2.is_empty());
            assert!(self.input.left_over.is_empty() && self.input.input.next().is_none());
            return Some((symbol1, ZIP_SPECIAL_EOF));
        }
        let symbol2 = symbol2.unwrap();
        assert!(seq2.len() > 0);

        let mut total_seq = seq1;
        total_seq.append(&mut seq2);
        self.create_symbol(&total_seq);
        Some((symbol1, symbol2))
    }

    fn find_symbol(&mut self) -> (Option<Symbol>, Vec<u8>) {
        let mut descent = self.trie.new_descent();

        let mut longest_symbol = None;
        let mut longest_seq = Vec::new();
        let mut extra_bytes = Vec::new();

        while let Some(byte) = self.input.next() {
            let (trie_entry, new_descent) = descent.down(byte);

            match trie_entry {
                Some(symbol) => {
                    // There is a symbol for this subsequence
                    longest_symbol = Some(*symbol);
                    longest_seq.append(&mut extra_bytes);
                    // Note: extra_bytes is now empty
                    longest_seq.push(byte);
                }
                None => {
                    // This subsequence doesn't have any symbol attached to it
                    // Remember the byte
                    extra_bytes.push(byte);
                }
            }

            match new_descent {
                Some(d) => {
                    descent = d;
                }
                None => {
                    // This descent is final
                    break;
                }
            }
        }
        // Either we reached the end of the input stream,
        // or the descent reached the end
        // Save the unused bytes and return the current symbol
        self.input.save_unused(extra_bytes);
        (longest_symbol, longest_seq)
    }
}

/// A wrapper over an iterator that can also save extra unused bytes
struct Input<I> {
    input:     I,
    left_over: Vec<u8> // last element must be the first to be returned
}

impl<I: Iterator<Item=u8>> Iterator for Input<I> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        self.left_over
            .pop()
            .or_else(|| self.input.next())
    }
}

impl<I> Input<I> {
    fn new(input: I) -> Self {
        Input {
            input,
            left_over: Vec::new()
        }
    }

    // The `unused` Vec is assumed to store the first received and unused
    // byte first
    // i.e. the client maintained a stack of receives bytes
    fn save_unused(&mut self, mut unused: Vec<u8>) {
        unused.reverse();
        self.left_over.append(&mut unused);
    }
}

/* DictDecoder */

pub struct DictDecoder<I> {
    encoded_stream: I,
    prev_decoded:   std::vec::IntoIter<u8>,
    mapping:        HashMap<Symbol, Vec<u8>>,
    abbrev:         Abbrev
}

impl<I: Iterator<Item=u8>> Iterator for DictDecoder<I> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        match self.prev_decoded.next() {
            Some(b) => {
                // Previously decoded sequence still has bytes
                return Some(b);
            }
            None => {
                // Figure out the next sequence (if any)
                let symbol1 = self.next_symbol()?;
                let symbol2 = self.next_symbol().unwrap();

                let seq1 = self.decode(symbol1);
                let seq2 = self.decode(symbol2);

                let mut total_seq = seq1.clone();
                total_seq.extend_from_slice(seq2);

                match self.next_code() {
                    Some(new_code) => {
                        self.new_mapping(new_code, total_seq.clone());
                    }
                    None => {
                        // nothing to do
                    }
                }

                self.store_seq(total_seq);
                self.next()
            }
        }
    }
}

impl<I: Iterator<Item=u8>> DictDecoder<I> {

    fn new(encoded_stream: I) -> Self {
        let mut decoder =
            DictDecoder {
                encoded_stream,
                prev_decoded: Vec::new().into_iter(),
                mapping:      HashMap::new(),
                abbrev:       Abbrev::start_after_trivial(),
            };

        // Set up the symbol mapping
        for b in 0x00..=0xff {
            decoder.new_mapping(b as u16, vec![b]);
        }
        // Add the special EOF sequence
        decoder.new_mapping(ZIP_SPECIAL_EOF, Vec::new());

        return decoder;
    }

    fn decode(&self, symbol: Symbol) -> &Vec<u8> {
        self.mapping
            .get(&symbol)
            .unwrap()
    }

    fn next_symbol(&mut self) -> Option<Symbol> {
        u16::deserialize(&mut self.encoded_stream)
    }

    fn store_seq(&mut self, seq: Vec<u8>) {
        self.prev_decoded = seq.into_iter();
    }

    fn next_code(&mut self) -> Option<Symbol> {
        self.abbrev.next()
    }

    fn new_mapping(&mut self, symbol: Symbol, seq: Vec<u8>) {
        self.mapping.insert(symbol, seq);
    }

}

/// A simple counter to generate the next unused symbol
struct Abbrev {
    counter: Symbol
}

impl Abbrev {
    fn new() -> Self {
        Abbrev {
            counter: 0
        }
    }

    fn start_after_trivial() -> Self {
        Abbrev {
            counter: 0x0100
        }
    }

    fn next(&mut self) -> Option<Symbol> {
        let symbol = self.counter;

        if symbol == ZIP_SPECIAL_EOF {
            None
        }
        else {
            self.counter += 1;
            Some(symbol)
        }
    }
}

/* TrieMap */

// Doesn't support mapping a value to the empty sequence
struct TrieMap<T>(Node<T>);

impl<T> TrieMap<T> {
    fn new() -> Self {
        TrieMap ( Node::new() )
    }

    fn new_descent(&self) -> Descent<T> {
        Descent::new(&self.0)
    }

    // Returns the previous value (if any) for that sequence
    fn insert(&mut self, seq: &[u8], value: T) -> Option<T> {
        // We don't support empty sequences
        assert!(seq.len() > 0);
        let mut curr_node = &mut self.0;
        for byte in &seq[0..seq.len()-1] {
            if !curr_node.has_child(*byte) {
                let prev_child = curr_node.add_child(*byte);
                assert!(prev_child.is_none());
            }
            curr_node = curr_node.child_mut(*byte).unwrap();
        }
        let final_node = curr_node;
        let last_byte = seq.last().unwrap();
        final_node.values
            .upsert(*last_byte, value)
    }

    fn print_stats(&self) {
        struct DFS<'a, U>(Vec<&'a Node<U>>);
        impl<'a, U> Iterator for DFS<'a, U> {
            type Item = &'a Node<U>;
            fn next(&mut self) -> Option<Self::Item> {
                let curr_node = self.0.pop()?;
                match &curr_node.children {
                    Content::Partial { keys: _, values } => {
                        for child in values.iter() {
                            self.0.push(child)
                        }
                    }
                    Content::Full(data) => {
                        for entry in data.iter() {
                            match entry.as_ref() {
                                Some(child) => self.0.push(child),
                                None => {},
                            }
                        }
                    }
                }
                Some(curr_node)
            }
        }

        let mut node_count = 0;
        let mut total_size = 0;
        let mut used_child_entries = 0;
        let mut tot_child_entries = 0;
        let mut used_value_entries = 0;
        let mut tot_value_entries = 0;
        for node in DFS(vec![&self.0]) {
            node_count += 1;

            total_size += std::mem::size_of::<Node<T>>();

            match &node.children {
                Content::Partial { keys: _, values } => {
                    used_child_entries += values.len();
                    tot_child_entries += values.len();
                }
                Content::Full(data) => {
                    used_child_entries += data.iter()
                                            .filter(|s| s.is_some())
                                            .count();
                    tot_child_entries += 256;
                }
            }

            match &node.values {
                Content::Partial { keys: _, values } => {
                    used_value_entries += values.len();
                    tot_value_entries += values.len();
                }
                Content::Full(data) => {
                    used_value_entries += data.iter()
                                            .filter(|s| s.is_some())
                                            .count();
                    tot_value_entries += 256;
                }
            }
        }
        println!("Trie stats:");
        println!("  Size of a single node: {}", ByteSize::b(std::mem::size_of::<Node<T>>() as u64));
        println!("  Size of a content entry: {}", ByteSize::b(std::mem::size_of::<Content<T>>() as u64));
        println!("  Node count: {}", node_count);
        println!("  Total size: {}", ByteSize::b(total_size as u64));
        let child_density = used_child_entries as f64 / tot_child_entries as f64;
        println!("  Child density: {:.2}%", child_density * 100.0);
        let value_density = used_value_entries as f64 / tot_value_entries as f64;
        println!("  Value density: {:.2}%", value_density * 100.0);
        let useful_child_size = used_child_entries * std::mem::size_of::<Node<T>>();
        let useful_value_size = used_value_entries * std::mem::size_of::<T>();
        println!("  Useful size: {}", ByteSize::b((useful_child_size + useful_value_size) as u64));
    }
}

impl<T> Drop for TrieMap<T> {
    fn drop(&mut self) {
        // Run a DFS over the Trie, dropping children nodes as we go along
        // This avoids a stack overflow in drop() when the trie is very deep
        // Because we have a recursive data structure, drop() is also called
        // recursively
        let mut stack: Vec<Box<Node<T>>> = Vec::new();

        fn handle_children<T>(curr_node: &mut Node<T>, stack: &mut Vec<Box<Node<T>>>) {
            match &mut curr_node.children {
                Content::Partial { keys: _, values } => {
                    stack.append(values)
                }
                Content::Full(data) => {
                    for child_ref in data.iter_mut() {
                        match std::mem::take(child_ref) {
                            Some(child_own) => {
                                stack.push(child_own)
                            }
                            None => {}
                        }
                    }
                }
            }
        }

        handle_children(&mut self.0, &mut stack);
        while !stack.is_empty() {
            let mut curr_node = stack.pop().unwrap();
            handle_children(&mut curr_node, &mut stack);
            // curr_node is dropped here
            // When it is dropped here, it doesn't have any children anymore,
            // avoiding the recursive call to drop()
        }
    }
}

// The two fields are independent:
// a given index may have a value but no child,
// and vice-versa
struct Node<T> {
    children: Content<Box<Node<T>>>,
    values:   Content<T>
}

// Optimize the memory requirement of the trie:
// While a node stores only a small number of entries,
// perform a linear scan on a stored 'keys' Vec
// When the node becomes big enough, revert to explicit
// format, which allows direct indexing
enum Content<T> {
    Partial {
        keys:   Vec<u8>,
        values: Vec<T>
    },
    Full (
        Box<[Option<T>; 256]>
    )
}

impl<T> Content<T> {
    fn new() -> Self {
        Self::Partial {
            keys:   Vec::new(),
            values: Vec::new()
        }
    }

    fn find(keys: &[u8], byte: u8) -> Option<usize> {
        keys.into_iter()
            .enumerate()
            .find(|(_, x)| **x == byte)
            .map(|(idx, _)| idx)
    }

    fn get(&self, byte: u8) -> Option<&T> {
        match self {
            Content::Partial { keys, values } => {
                let idx = Self::find(keys, byte)?;
                values.get(idx)
            }
            Content::Full(data) => {
                (&data[byte as usize]).as_ref()
            }
        }
    }

    fn get_mut(&mut self, byte: u8) -> Option<&mut T> {
        match self {
            Content::Partial { keys, values } => {
                let idx = Self::find(keys, byte)?;
                values.get_mut(idx)
            }
            Content::Full(data) => {
                (&mut data[byte as usize]).as_mut()
            }
        }
    }

    // When is a Partial node big enough to be converted to Full
    const THRESHOLD: usize = 64;

    fn upsert(&mut self, byte: u8, new_value: T) -> Option<T> {
        match self {
            Content::Partial { keys, values } => {
                match Self::find(keys, byte) {
                    Some(existing_idx) => {
                        // Replace
                        let target = values.get_mut(existing_idx).unwrap();
                        let prev_value = std::mem::replace(target, new_value);
                        Some(prev_value)
                    }
                    None => {
                        // Insert
                        if keys.len() < Self::THRESHOLD {
                            // Remain in partial format
                            keys.push(byte);
                            values.push(new_value);
                            None
                        }
                        else {
                            // Convert to Full
                            self.to_full();
                            self.upsert(byte, new_value)
                        }
                    }
                }
            }
            Content::Full(data) => {
                std::mem::replace(&mut data[byte as usize], Some(new_value))
            }
        }
    }

    fn to_full(&mut self) {
        println!("Converting Content::Partial into Content::Full!");
        let (partial_keys, partial_values) =
            match self {
                Content::Partial { keys, values } =>
                    (std::mem::take(keys), std::mem::take(values)),
                _ => unreachable!(),
            };

        let mut full_data = default_array();
        for (byte, value) in partial_keys.into_iter()
                                        .zip(partial_values.into_iter()) {
            full_data[byte as usize] = Some(value);
        }

        *self = Content::Full(Box::from(full_data));
    }
}

impl<T> Node<T> {
    fn has_child(&self, byte: u8) -> bool {
        self.child(byte).is_some()
    }

    fn child(&self, byte: u8) -> Option<&Node<T>> {
        self.children
            .get(byte)
            .map(Box::as_ref)
    }

    // Returns the previous entry (if any)
    fn add_child(&mut self, byte: u8) -> Option<Box<Node<T>>> {
        let new_child = Self::new();
        let new_child = Box::from(new_child);
        self.children
            .upsert(byte, new_child)
    }

    fn new() -> Self {
        Node {
            children: Content::new(),
            values:   Content::new()
        }
    }

    fn child_mut(&mut self, byte: u8) -> Option<&mut Self> {
        self.children
            .get_mut(byte)
            .map(Box::as_mut)
    }

    fn value_of(&self, byte: u8) -> Option<&T> {
        self.values.get(byte)
    }
}

/// An immutable descent into the [TrieMap]
struct Descent<'a, T> {
    curr_node: &'a Node<T>
}

impl<'a, T> Descent<'a, T> {
    fn new(root: &'a Node<T>) -> Self {
        Descent {
            curr_node: root
        }
    }

    fn down(self, byte: u8) -> (Option<&'a T>, Option<Self>) {
        let value = self.curr_node.value_of(byte);
        let new_self = self.curr_node.child(byte).map(Self::new);
        (value, new_self)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn test_encoding(input: &[u8], expected_output: &[u16]) {
        let mut zip_output = Vec::new();
        zip_dict_encode(input.into_iter().cloned(), &mut zip_output)
            .unwrap();

        // Convert the zip output into a sequence of u16
        let mut output_bytes = zip_output.into_iter();
        let mut zip_symbols = Vec::new();
        while let Some(symbol) = u16::deserialize(&mut output_bytes) {
            zip_symbols.push(symbol);
        }

        assert_eq!(&zip_symbols, expected_output);
    }

    #[test]
    fn enc0() {
        test_encoding(&[], &[])
    }

    #[test]
    fn enc1() {
        test_encoding(&[1], &[1, ZIP_SPECIAL_EOF])
    }

    #[test]
    fn enc2() {
        test_encoding(&[1, 2], &[1, 2])
    }

    #[test]
    fn enc4() {
        test_encoding(&[1, 2, 1, 3], &[1, 2, 1, 3])
    }

    #[test]
    fn enc6() {
        test_encoding(&[1, 2, 1, 2, 1, 2], &[1, 2, 0x0100, 0x0100])
    }

    fn test_decoding(input: &[u8]) {
        let mut zip_encoded = Vec::new();
        zip_dict_encode(input.into_iter().cloned(), &mut zip_encoded)
            .unwrap();

        // Decode
        let zip_decoded: Vec<u8> = zip_dict_decode(zip_encoded.into_iter())
                                       .collect();

        assert_eq!(&zip_decoded, input);
    }

    #[test]
    fn dec0() {
        test_decoding(&[])
    }

    #[test]
    fn dec1() {
        test_decoding(&[1])
    }

    #[test]
    fn dec2() {
        test_decoding(&[1, 2])
    }

    #[test]
    fn dec4() {
        test_decoding(&[1, 2, 1, 3])
    }

    #[test]
    fn dec6() {
        test_decoding(&[1, 2, 1, 2, 1, 2])
    }
}
