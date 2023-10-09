#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Bit {
    Zero,
    One
}

impl Bit {
    pub fn byte(&self) -> u8 {
        self.into()
    }
}

impl From<&Bit> for u8 {
    fn from(b: &Bit) -> u8 {
        match b {
            Bit::Zero => 0,
            Bit::One  => 1,
        }
    }
}

pub fn push_bit(n: &mut u8, b: Bit) {
    *n = (*n << 1) | b.byte();
}

pub fn byte_from_slice(bits: &[Bit]) -> Option<u8> {
    if bits.len() != 8 {
        return None
    }

    let mut n = 0u8;
    for b in bits {
        push_bit(&mut n, *b);
    }

    return Some(n);
}

pub fn nth(byte: u8, idx: u8) -> Bit {
    let masked = byte & (1u8 << idx);
    if masked == 0 {
        Bit::Zero
    }
    else {
        Bit::One
    }
}

pub fn bit_array(byte: u8) -> [Bit; 8] {
    [
        nth(byte, 0),
        nth(byte, 1),
        nth(byte, 2),
        nth(byte, 3),

        nth(byte, 4),
        nth(byte, 5),
        nth(byte, 6),
        nth(byte, 7),
    ]
}

pub trait WriteBit {
    fn write(&mut self, b: Bit);

    fn write_byte(&mut self, n: u8) {
        for b in bit_array(n).iter() {
            self.write(*b);
        }
    }

    fn pad_and_flush(&mut self);
}

pub struct IoBitWriter<W> {
    writer:    W,
    curr_bits: u8,
    bit_count: u8
}

impl<W> IoBitWriter<W> {
    pub fn new(writer: W) -> Self {
        IoBitWriter {
            writer,
            curr_bits: 0,
            bit_count: 0
        }
    }

    pub fn into_inner(self) -> W {
        self.writer
    }
}

impl<W: std::io::Write> WriteBit for IoBitWriter<W> {
    fn write(&mut self, b: Bit) {
        push_bit(&mut self.curr_bits, b);

        self.bit_count += 1;
        if self.bit_count == 8 {
            self.writer.write_all(&[self.curr_bits]);
            self.bit_count = 0;
        }
    }

    fn write_byte(&mut self, n: u8) {
        if self.bit_count == 0 {
            self.writer.write_all(&[n]);
        }
        else {
            let msb = self.curr_bits << (8 - self.bit_count);
            let lsb = n >> self.bit_count;
            let completed_byte = msb & lsb;
            self.writer.write_all(&[completed_byte]);

            let mut mask = 0u8;
            for _ in 0..self.curr_bits {
                mask = (mask << 1) & 1;
            }
            self.curr_bits = n & mask;
        }
    }

    #[allow(unused_parens)]
    fn pad_and_flush(&mut self) {
        if self.bit_count != 0 {
            // Pad the remains with 0s
            self.curr_bits <<= (8 - self.bit_count);
            self.writer.write_all(&[self.curr_bits]);

            self.curr_bits = 0;
            self.bit_count = 0;
        }
        self.writer.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::{Bit, IoBitWriter, WriteBit};

    #[test]
    fn byte_of_b0() {
        assert_eq!(Bit::Zero.byte(), 0u8);
    }

    #[test]
    fn byte_of_b1() {
        assert_eq!(Bit::One.byte(), 1u8);
    }

    fn vec_bw() -> IoBitWriter<Vec<u8>> {
        IoBitWriter::new(Vec::new())
    }

    #[test]
    fn write_x00() {
        let mut bw = vec_bw();
        for _ in 0..8 {
            bw.write(Bit::Zero);
        }
        assert_eq!(bw.into_inner(), vec![0]);
    }

    #[test]
    fn write_xff() {
        let mut bw = vec_bw();
        for _ in 0..8 {
            bw.write(Bit::One);
        }
        assert_eq!(bw.into_inner(), vec![0xffu8]);
    }
}
