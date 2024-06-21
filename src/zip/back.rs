use std::{cmp::max, collections::VecDeque, io};
use crate::{ser::{Serialize, Deserialize}, utils::default_array};

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

#[allow(dead_code)]
pub fn zip_back_decode<I: Iterator<Item=u8>>(input: I) -> Decoder<I> {
    Decoder::new(input)
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

impl Symbol {
    const ENUM_CODE_EXPLICIT: u8 = 0;
    const ENUM_CODE_LOOKBACK: u8 = 1;

    fn mask(msb: u8) -> Len {
        (msb as Len).rotate_right(1)
    }

    fn compress_len(len: usize, enum_decider: u8) -> Len {
        assert!(len <= (Len::MAX >> 1) as usize);

        let mask: Len = Self::mask(enum_decider);
        (len as Len) | mask
    }

    fn decompress_len(compressed_len: Len) -> (u8, usize) {
        let mask: Len = Self::mask(1);
        let msb = (compressed_len & mask).rotate_left(1) as u8;

        let mask = !mask;
        let lsbs = compressed_len & mask;

        (msb, lsbs as usize)
    }
}


impl Serialize for Symbol {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        match self {
            Symbol::Explicit(explicit_data) => {
                let compacted_len = Self::compress_len(explicit_data.len(), Self::ENUM_CODE_EXPLICIT);
                compacted_len.serialize(writer)?;

                // Note: we can't use Vec::serialize() as it also serializes
                // the length as a usize
                for b in explicit_data.iter() {
                    b.serialize(writer)?;
                }
            }
            Symbol::LookBack(LookBack { len, back }) => {
                Self::compress_len(*len, Self::ENUM_CODE_LOOKBACK)
                    .serialize(writer)?;

                (*back as Back).serialize(writer)?;
            }
        }
        Ok(())
    }
}

impl Deserialize for Symbol {
    fn deserialize<I: Iterator<Item = u8>>(stream: &mut I) -> Option<Self> {
        let compressed_len = Len::deserialize(stream)?;
        let (enum_decider, len) = Self::decompress_len(compressed_len);
        match enum_decider {
            Self::ENUM_CODE_EXPLICIT => {
                let data: Vec<u8> = stream.take(len).collect();
                assert!(data.len() == len);
                Some(Symbol::Explicit(data))
            }
            Self::ENUM_CODE_LOOKBACK => {
                let back = Back::deserialize(stream)?;
                Some(Symbol::LookBack(LookBack { len, back: back as usize }))
            }
            _ => unreachable!(),
        }
    }
}

/* Encoder */

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
                            push_explicit(curr_explicit, channel);
                            return;
                        }
                    }
                }
                None => {
                    // No repetition: extend the current explicit
                    // TODO consider using a goto or conditional guard on the match arm to share the code with above
                    let must_return = extend_explicit(self, &mut curr_explicit);
                    if must_return {
                        push_explicit(curr_explicit, channel);
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
                let rel_pos = starting_index - self.start;
                let subseq =
                    self.data
                        .iter()
                        .cloned()
                        .skip(rel_pos);
                let lookback = self.data.len() - rel_pos;
                (lookback, subseq)
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

    fn duplicate(&mut self, lb: &LookBack) -> Vec<u8> {
        let bytes_to_copy: Vec<u8> =
            self.data
                .iter()
                .cloned()
                .skip(self.data.len() - lb.back)
                .take(lb.len)
                .collect();
        self.write_all(&bytes_to_copy);
        bytes_to_copy
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
    }
}

/* Decoder */

// TODO try to share some of this with the zip dict decoder
pub struct Decoder<I> {
    iter:    I,
    history: History,
    decoded: VecDeque<u8>
}

impl<I: Iterator<Item = u8>> Iterator for Decoder<I> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        self.decoded
            .pop_front()
            .or_else(|| {
                self.decode_next();
                self.decoded.pop_front()
            })
    }
}

impl<I> Decoder<I> {
    fn new(input: I) -> Self {
        Decoder {
            iter:    input,
            history: History::new(),
            decoded: VecDeque::new(),
        }
    }
}

impl<I: Iterator<Item = u8>> Decoder<I> {
    fn decode_next(&mut self) {
        let symbol = Symbol::deserialize(&mut self.iter);
        match symbol {
            Some(Symbol::Explicit(explicit_data)) => {
                self.history.write_all(&explicit_data);
                self.decoded.extend(&explicit_data);
            }
            Some(Symbol::LookBack(lb)) => {
                let lookback_slice = self.history.duplicate(&lb);
                self.decoded.extend(lookback_slice);
            }
            None => {
                // input stream is empty, nothing left to do
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn test_encode(input: &[u8], expected_symbols: &[Symbol]) {
        let mut zip_output = Vec::new();
        zip_back_encode(input.iter().cloned(), &mut zip_output)
            .unwrap();

        let mut expected_output = Vec::new();
        for s in expected_symbols {
            s.serialize(&mut expected_output)
                .unwrap();
        }

        assert_eq!(&zip_output, &expected_output);
    }

    #[test]
    fn enc0() {
        test_encode(&[], &[])
    }

    #[test]
    fn enc1() {
        test_encode(&[0x01], &[Symbol::Explicit(vec![0x01])])
    }

    #[test]
    fn enc2_no_repeat() {
        test_encode(&[0x01, 0x02], &[Symbol::Explicit(vec![0x01, 0x02])])
    }

    #[test]
    fn enc2_repeat() {
        test_encode(&[0x01, 0x01], &[Symbol::Explicit(vec![0x01, 0x01])])
    }

    #[test]
    fn enc6() {
        test_encode(&[0x01; 6], &[Symbol::Explicit(Vec::from([0x01; 6]))])
    }

    #[test]
    fn enc16_repeat() {
        test_encode(&[0x01; 16],
            &[Symbol::Explicit(Vec::from([0x01; 8])),
                        Symbol::LookBack(LookBack { len: 8, back: 8 })])
    }

    #[test]
    fn enc16_no_repeat() {
        let data =
            [
                0x01, 0x01, 0x01, 0x01,
                0x01, 0x01, 0x01, 0x01,
                0x02, 0x02, 0x02, 0x02,
                0x02, 0x02, 0x02, 0x02
            ];

        test_encode(
            &data,
            &[Symbol::Explicit(Vec::from(data))])
    }

    fn test_decode(input: &[u8]) {
        let mut zip_output = Vec::new();
        zip_back_encode(input.iter().cloned(), &mut zip_output)
            .unwrap();

        let decoded: Vec<u8> = zip_back_decode(zip_output.into_iter()).collect();

        assert_eq!(&decoded, &input);
    }


    #[test]
    fn dec0() {
        test_decode(&[])
    }

    #[test]
    fn dec1() {
        test_decode(&[0x01])
    }

    #[test]
    fn dec2_no_repeat() {
        test_decode(&[0x01, 0x02])
    }

    #[test]
    fn dec2_repeat() {
        test_decode(&[0x01, 0x01])
    }

    #[test]
    fn dec6() {
        test_decode(&[0x01; 6])
    }

    #[test]
    fn dec16_repeat() {
        test_decode(&[0x01; 16])
    }

    #[test]
    fn dec16_no_repeat() {
        let data =
            [
                0x01, 0x01, 0x01, 0x01,
                0x01, 0x01, 0x01, 0x01,
                0x02, 0x02, 0x02, 0x02,
                0x02, 0x02, 0x02, 0x02
            ];

        test_decode(&data)
    }

}
