use std::{cmp::max, collections::VecDeque, io};
use crate::{ser::Serialize, utils::default_array};

#[allow(dead_code)]
pub fn zip_back_encode<I: Iterator<Item=u8>, W: io::Write>(bytes: I, output: &mut W) -> io::Result<()> {
    let mut encoder = Encoder::new(bytes);
    let mut channel = Vec::new();
    encoder.next_symbols(&mut channel);
    while !channel.is_empty() {
        for s in channel.drain(..) {
            s.serialize(output)?;
        }
        encoder.next_symbols(&mut channel);
    }
    Ok(())
}

enum Symbol {
    Explicit(Vec<u8>),
    LookBack(LookBack)
}

struct LookBack {
    len:  usize,
    back: usize
}

type Len = u16;
type Back = u16;

const ENUM_CODE_EXPLICIT: u8 = 0;
const ENUM_CODE_LOOKBACK: u8 = 1;

impl Serialize for Symbol {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        fn form_length(len: usize, enum_decider: u8) -> Len {
            assert!(len <= (Len::MAX >> 1) as usize);

            let mask: Len = (enum_decider as Len).rotate_right(1);
            (len as Len) | mask
        }

        match self {
            Symbol::Explicit(explicit_data) => {
                let compacted_len = form_length(explicit_data.len(), ENUM_CODE_EXPLICIT);
                compacted_len.serialize(writer)?;

                // Note: we can't use Vec::serialize() as it also serializes
                // the length as a usize
                for b in explicit_data.iter() {
                    b.serialize(writer)?;
                }
            }
            Symbol::LookBack(LookBack { len, back }) => {
                form_length(*len, ENUM_CODE_LOOKBACK)
                    .serialize(writer)?;

                (*back as Back).serialize(writer)?;
            }
        }
        Ok(())
    }
}

struct Encoder<I> {
    history: History,
    input:   Buffered<I, u8>
}

impl<I> Encoder<I> {
    fn new(input: I) -> Self {
        Encoder {
            history: History::new(),
            input:   Buffered::new(input)
        }
    }
}

// TODO explain
const MIN_REP: Len = 6;

impl<I: Iterator<Item = u8>> Encoder<I> {
    fn next_symbols(&mut self, channel: &mut Vec<Symbol>) {
        let mut curr_explicit: Vec<u8> = Vec::new();

        fn push_explicit(curr_explicit: Vec<u8>, channel: &mut Vec<Symbol>) {
            if !curr_explicit.is_empty() {
                channel.push(Symbol::Explicit(curr_explicit));
            }
        }

        fn push_lookback<I: Iterator<Item = u8>>(encoder: &mut Encoder<I>, lb: LookBack, channel: &mut Vec<Symbol>) {
            encoder.commit_bytes(lb.len);
            channel.push(Symbol::LookBack(lb));
        }

        // Returns whether the input sequence is depleted
        fn extend_explicit<I: Iterator<Item = u8>>(encoder: &mut Encoder<I>, curr_explicit: &mut Vec<u8>) -> bool {
            // No repetition was found
            // Restore the input
            encoder.input.restore();

            // We always try to double the current explicit size
            let try_to_get = max(curr_explicit.len(), 2);
            let mut new_bytes: Vec<u8> =
                (&mut encoder.input)
                        .take(try_to_get)
                        .collect();

            // Accept the bytes that are now part of the explicit
            encoder.input.accept(new_bytes.len());

            // Remember the bytes in the lookback data structure
            encoder.history.write_all(&new_bytes);

            let input_depleted = new_bytes.len() < try_to_get;

            curr_explicit.append(&mut new_bytes);

            return input_depleted;
        }

        loop {
            match self.next_repetition() {
                Some(lb@LookBack { len: rep_len, .. }) => {
                    // TODO explain that '>=' is better than '>'
                    if rep_len >= MIN_REP as usize {
                        // Repetition is useful because long enough
                        // Push the current explicit symbol
                        // and the new repetition
                        push_explicit(curr_explicit, channel);
                        push_lookback(self, lb, channel);
                        return;
                    }
                    else {
                        // Repetition is too short
                        // Extend the current explicit
                        let must_return = extend_explicit(self, &mut curr_explicit);
                        if must_return {
                            return;
                        }
                    }
                }
                None => {
                    // No repetition: extend the current explicit
                    // TODO consider using a goto or conditional guard on the match arm to share the code with above
                    let must_return = extend_explicit(self, &mut curr_explicit);
                    if must_return {
                        return;
                    }
                }
            }
        }
    }

    fn next_repetition(&mut self) -> Option<LookBack> {
        let mut checkpoint = Buffered::new(&mut self.input);
        let starting_byte = checkpoint.next()?;
        checkpoint.restore();

        let mut max_rep_len = 0;
        let mut max_rep_back = 0;
        for (lookback, subseq) in self.history.subseqs_starting(starting_byte) {
            let rep_len =
                (&mut checkpoint).zip(subseq)
                    .take_while(|(a, b)| a == b)
                    .count();

            if rep_len > max_rep_len {
                max_rep_len = rep_len;
                max_rep_back = lookback;
            }
            checkpoint.restore();
        }
        Some(LookBack { len: max_rep_len, back: max_rep_back })
    }

    fn commit_bytes(&mut self, nbytes: usize) {
        self.input.restore();
        for _i in 0..nbytes {
            let b = self.input.next().unwrap();
            self.history.write(b);
        }
        self.input.accept(nbytes);
    }
}

struct History {
    data:  VecDeque<u8>,
    index: [Vec<usize>; 256],
    /// The position of the ring buffer's head in the entire input stream
    start: usize
}

const MAX_RING_BUFFER_SIZE: usize = (Back::MAX as usize) + 1;

impl History {
    fn new() -> Self {
        History {
            data:  VecDeque::with_capacity(MAX_RING_BUFFER_SIZE),
            index: default_array(),
            start: 0
        }
    }

    fn subseqs_starting(&self, starting_byte: u8) -> impl Iterator<Item = (usize, impl Iterator<Item = u8> + '_)> {
        self.index[starting_byte as usize]
            .iter()
            .map(|starting_index| {
                let subseq =
                    self.data
                        .iter()
                        .cloned()
                        .skip(starting_index - self.start);
                (*starting_index, subseq)
            })
    }

    fn write(&mut self, b: u8) {
        // Is our ring buffer full?
        if self.data.len() == MAX_RING_BUFFER_SIZE {
            // Free up a slot
            self.data.pop_front();
            // Increase the start value
            self.start += 1;
        }

        // Write the byte in the ring buffer
        self.data.push_back(b);

        // Update the index
        self.update_index(b);
    }

    fn update_index(&mut self, b: u8) {
        let pos = self.start + self.data.len() - 1;
        let index_entry = &mut self.index[b as usize];

        // 1. Try to find an entry to update
        for m in index_entry.iter_mut() {
            if *m < self.start {
                // Entry is no longer relevant
                // We can replace it
                *m = pos;
                return;
            }
        }
        // No entry could be replace
        // Add a new one now
        index_entry.push(pos);
    }

    fn write_all(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.write(*b);
        }
    }
}

struct Buffered<I, T> {
    iter:     I,
    saved:    VecDeque<T>,
    read_pos: usize
}

impl<I, T> Buffered<I, T> {
    fn new(iter: I) -> Self {
        Buffered {
            iter,
            saved: VecDeque::new(),
            read_pos: 0
        }
    }

    fn restore(&mut self) {
        self.read_pos = 0;
    }

    fn accept(&mut self, n: usize) {
        assert!(self.saved.len() >= n);
        for _i in 0..n {
            self.saved.pop_front();
        }
        self.read_pos -= n;
    }

    fn can_read_saved(&self) -> bool {
        self.read_pos < self.saved.len()
    }

    fn next_saved(&mut self) -> Option<&T> {
        let read_pos = self.read_pos;
        if self.can_read_saved() {
            self.read_pos += 1;
        }
        self.saved.get(read_pos)
    }
}

impl<I: Iterator<Item = T>, T> Buffered<I, T> {
    fn read_and_save(&mut self) {
        match self.iter.next() {
            Some(x) => self.saved.push_back(x),
            None => {},
        }
    }
}

impl<I: Iterator<Item = T>, T: Copy> Iterator for Buffered<I, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.can_read_saved() {
            self.read_and_save();
        }
        self.next_saved()
            .map(|x| *x)

        /*let mut res = self.saved.get(self.read_pos);
        if res.is_none() {
            // Try to read from the iterator
            self.iter
                .next()
                .into_iter()
                .for_each(|x| {
                    self.saved.push(x);
                    self.read_pos += 1;
                });

            res = self.saved.get(self.read_pos);
        }

        return res;*/

        /*match self.read_pos {
            Some(pos) => {
                // Read from the saved data
                let x = self.saved[pos];
                pos += 1;
                return Some(x);
            }
            None => {
                // Read from the iterator
                match self.iter.next() {
                    Some(x) => {
                        self.saved.push(x);
                        return Some(x);
                    }
                    None => {
                        return None;
                    }
                }
            }
        }*/
    }
}
/*
struct Checkpoint<I, T> {
    iter:     I,
    saved:    Vec<T>,
    read_pos: usize
}

impl<I, T> Checkpoint<I, T> {
    fn from(iter: I) -> Self {
        Checkpoint {
            iter,
            saved: Vec::new(),
            read_pos: 0
        }
    }

    fn restore(&mut self) {
        self.read_pos = 0;
    }
}

impl<I: Iterator<Item = T>, T> Iterator for Checkpoint<I, T> {
    type Item = &T;

    fn next(&mut self) -> Option<Self::Item> {
        let mut res = self.saved.get(self.read_pos);
        if res.is_none() {
            // Try to read from the iterator
            self.iter
                .next()
                .into_iter()
                .for_each(|x| {
                    self.saved.push(x);
                    self.read_pos += 1;
                });

            res = self.saved.get(self.read_pos);
        }

        return res;

        /*match self.read_pos {
            Some(pos) => {
                // Read from the saved data
                let x = self.saved[pos];
                pos += 1;
                return Some(x);
            }
            None => {
                // Read from the iterator
                match self.iter.next() {
                    Some(x) => {
                        self.saved.push(x);
                        return Some(x);
                    }
                    None => {
                        return None;
                    }
                }
            }
        }*/
    }
}*/
