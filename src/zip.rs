use std::{io, slice};
use crate::ser::Serialize;

const ZIP_SPECIAL_EOF: Symbol = Symbol::MAX;

#[allow(dead_code)]
pub fn zip_dict<I: Iterator<Item=u8>, W: io::Write>(bytes: I, output: &mut W) -> io::Result<()> {
    let mut zip = ZipDict::new(bytes);
    let mut zipped = zip.next_triplet();

    loop {
        match zipped {
            Encoded::Triplet(triplet) => {
                for symbol in triplet {
                    symbol.serialize(output)?;
                }
            }
            Encoded::PartialEOF(single_symbol) => {
                ZIP_SPECIAL_EOF.serialize(output)?;
                single_symbol.serialize(output)?;
            }
            Encoded::CleanEOF => {
                break;
            }
        }
        zipped = zip.next_triplet();
    }

    Ok(())
}

struct ZipDict<I> {
    input:   Input<I>,
    trie:    TrieMap<Symbol>,
    counter: Symbol
}

type Symbol = u16;

enum Encoded {
    Triplet([Symbol; 3]),
    PartialEOF(Symbol),
    CleanEOF
}

impl<I> ZipDict<I> {
    fn new(input: I) -> Self {
        let mut zip = ZipDict {
            input:   Input::new(input),
            trie:    TrieMap::new(),
            counter: 0,
        };

        // Important: use an inclusive range
        for b in 0x00..=0xff {
            zip.new_symbol(slice::from_ref(&b));
        }
        assert_eq!(zip.counter, 0x0100);

        return zip;
    }

    fn new_symbol(&mut self, seq: &[u8]) -> Symbol {
        let symbol = self.counter;
        assert!(symbol != ZIP_SPECIAL_EOF);
        self.counter += 1;
        self.trie.insert(seq, symbol);
        return symbol;
    }
}

impl<I: Iterator<Item=u8>> ZipDict<I> {
    fn next_triplet(&mut self) -> Encoded {
        let (symbol1, seq1) = self.find_symbol();

        if symbol1.is_none() {
            // That can only happen if the input stream is empty
            // 1 byte always yields a symbol
            assert!(seq1.is_empty());
            assert!(self.input.left_over.is_empty() && self.input.input.next().is_none());
            return Encoded::CleanEOF;
        }
        let symbol1 = symbol1.unwrap();
        assert!(seq1.len() > 0);

        let (symbol2, mut seq2) = self.find_symbol();

        if symbol2.is_none() {
            // Here too, this can only happen if the input stream is empty
            assert!(seq2.is_empty());
            assert!(self.input.left_over.is_empty() && self.input.input.next().is_none());
            return Encoded::PartialEOF(symbol1);
        }
        let symbol2 = symbol2.unwrap();
        assert!(seq2.len() > 0);

        let mut total_seq = seq1;
        total_seq.append(&mut seq2);
        let new_symbol = self.new_symbol(&total_seq);
        Encoded::Triplet([symbol1, symbol2, new_symbol])
    }

    fn find_symbol(&mut self) -> (Option<Symbol>, Vec<u8>) {
        let mut descent = self.trie.new_descent();

        let mut longest_symbol = None;
        let mut longest_seq = Vec::new();
        let mut extra_bytes = Vec::new();

        // while let Some(byte) = self.input.next() {
        //     match descent.down(byte) {
        //         Some(trie_entry) => {
        //             match trie_entry.as_ref() {
        //                 Some(symbol) => {
        //                     // There is a symbol for this subsequence
        //                     longest_symbol = Some(*symbol);
        //                     longest_seq.append(&mut extra_bytes);
        //                     // Note: ewtra_bytes is now empty
        //                 }
        //                 None => {
        //                     // This subsequence doesn't have any symbol attached to it
        //                     // Remember the byte
        //                     extra_bytes.push(byte);
        //                 }
        //             }
        //         }
        //         None => {
        //             // This descent is final
        //             // Keep track of the extra bytes
        //             self.input.save_unused(extra_bytes);
        //             // Return the current symbol
        //             return (longest_symbol, longest_seq);
        //         }
        //     }
        // }

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

struct Input<I> {
    input:     I,
    left_over: Vec<u8> // last element must be the first to be returned
}

impl<I: Iterator<Item=u8>> Iterator for Input<I> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        self.left_over.pop()
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

// Doesn't support mapping a value to the empty sequence
struct TrieMap<T>(Node<T>);

impl<T> TrieMap<T> {
    fn new() -> Self {
        TrieMap ( Node::new() )
    }

    fn new_descent(&self) -> Descent<T> {
        Descent::new(&self.0)
    }

    /*fn new_descent_mut(&mut self) -> DescentMut<T> {
        DescentMut::new(&mut self.0)
    }*/

    // Returns the previous value (if any) for that sequence
    fn insert(&mut self, seq: &[u8], value: T) -> Option<T> {
        // We don't support empty sequences
        assert!(seq.len() > 0);
        let mut curr_node = &mut self.0;
        for byte in &seq[0..seq.len()-1] {
            if !curr_node.has_child(*byte) {
                curr_node.add_child(*byte);
            }
            curr_node = curr_node.child_mut(*byte).unwrap();
        }
        let final_node = curr_node;
        let last_byte = seq.last().unwrap();
        let previous_value = std::mem::replace(&mut final_node.values[*last_byte as usize], Some(value));
        return previous_value;
        // let mut descent = self.new_descent_mut();
        // for byte in &seq[0..seq.len()-1] {
        //     descent.down_or_create_empty(*byte);
        // }
        // let final_node = descent.curr_node;
        // let last_byte = seq.last().unwrap(); // note: we don't support empty sequences
        // let previous_value = std::mem::replace(&mut final_node.values[*last_byte as usize], Some(value));
        // return previous_value;
    }
}

// The two fields are independent:
// a given index may have a value but no child,
// and vice-versa
struct Node<T> {
    children: [Option<Box<Node<T>>>; 256],
    values:   [Option<T>; 256]
}

impl<T> Node<T> {
    fn has_child(&self, byte: u8) -> bool {
        self.child(byte).is_some()
    }

    fn child(&self, byte: u8) -> Option<&Node<T>> {
        self.children[byte as usize]
            .as_ref()
            .map(Box::as_ref)
    }

    // Returns the previous entry (if any)
    fn add_child(&mut self, byte: u8) -> Option<Box<Node<T>>> {
        let new_child = Self::new();
        let new_child = Some(Box::from(new_child));
        std::mem::replace(&mut self.children[byte as usize], new_child)
    }

    fn new() -> Self {
        Node {
            children: default_array(),
            values:   default_array()
        }
    }

    fn child_mut(&mut self, byte: u8) -> Option<&mut Self> {
        self.children[byte as usize]
            .as_mut()
            .map(Box::as_mut)
    }

    fn value_of(&self, byte: u8) -> Option<&T> {
        self.values[byte as usize]
            .as_ref()
    }
}

fn default_array<const N: usize, T: Default>() -> [T; N] {
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

struct Descent<'a, T> {
    // nodes: Vec<&'a mut Node<T>>  // TODO do we ever need to keep a stack here? can't we just keep track of the last one?
    curr_node: &'a Node<T>
}

impl<'a, T> Descent<'a, T> {
    fn new(root: &'a Node<T>) -> Self {
        Descent {
            curr_node: root
            // nodes: vec![root]
        }
    }

    fn down(self, byte: u8) -> (Option<&'a T>, Option<Self>) {
        // let curr_node = *self.nodes.last()?;
        // FIXME we want that value even if we didn't go down
        let value = self.curr_node.value_of(byte);
        let new_self = self.curr_node.child(byte).map(Self::new);
        (value, new_self)
        // let value = &mut self.curr_node.values[byte as usize];
        // match self.curr_node.children[byte as usize].as_mut() {
        //     Some(mut bx) => {
        //         self.curr_node = bx.as_mut();
        //         // let child = bx.as_mut();
        //         // self.nodes.push(child);
        //         Some(value)
        //     }
        //     None => {
        //         None
        //     }
        // }
    }
}

// Note: the borrow checker doesn't like this implementation at all
/*struct DescentMut<'a, T> {
    // nodes: Vec<&'a mut Node<T>>  // TODO do we ever need to keep a stack here? can't we just keep track of the last one?
    curr_node: &'a mut Node<T>
}

impl<'a, T> DescentMut<'a, T> {
    fn new(root: &'a mut Node<T>) -> Self {
        DescentMut {
            curr_node: root
            // nodes: vec![root]
        }
    }

    fn down_or_create_empty(&'a mut self, byte: u8) {
        if !self.curr_node.has_child(byte) {
            self.curr_node.add_child(byte);
        }
        self.curr_node = self.curr_node.child_mut(byte).unwrap();
    }
}
*/

#[cfg(test)]
mod test {
    use super::*;
    use crate::ser::Deserialize;

    fn test_encoding(input: &[u8], expected_output: &[u16]) {
        let mut zip_output = Vec::new();
        zip_dict(input.into_iter().cloned(), &mut zip_output)
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
        test_encoding(&[1], &[ZIP_SPECIAL_EOF, 1])
    }

    #[test]
    fn enc2() {
        test_encoding(&[1, 2], &[1, 2, 0x0100])
    }

    #[test]
    fn enc4() {
        test_encoding(&[1, 2, 1, 3], &[1, 2, 0x0100, 1, 3, 0x0101])
    }

    #[test]
    fn enc6() {
        test_encoding(&[1, 2, 1, 2, 1, 2], &[1, 2, 0x0100, 0x0100, 0x0100, 0x0101])
    }
}
