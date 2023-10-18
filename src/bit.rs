use std::io;

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

#[derive(Clone, Copy)]
pub enum BitOrder {
    /* Most Significant Bit first
     * Bit indexes:
     *   0123 4567
     */
    MsbFirst,
    /* Least Significant Bit first
     * Bit indexes:
     *   7654 3210
     */
    LsbFirst,
}

pub fn push_bit(n: &mut u8, b: Bit, bo: BitOrder) {
    match bo {
        /* If you're using bytes in Msb order it makes sense to do this:
         *
         * 0123 4567 << x
         * 1234 567x
         */
        BitOrder::MsbFirst => {
            *n = (*n << 1) | b.byte();
        }
        // The logic here is the opposite of the above case
        BitOrder::LsbFirst => {
            *n = (*n >> 1) | (b.byte() << 7);
        }
    }
}

pub fn byte_from_slice(bits: &[Bit], bo: BitOrder) -> Option<u8> {
    if bits.len() != 8 {
        return None
    }

    let mut n = 0u8;
    for b in bits {
        push_bit(&mut n, *b, bo);
    }

    return Some(n);
}

#[allow(unused_parens)]
pub fn nth(byte: u8, idx: u8, bo: BitOrder) -> Bit {
    let mask =
        match bo {
            // 7654 3210
            BitOrder::LsbFirst => (1u8 << idx),
            // 0123 4567
            BitOrder::MsbFirst => (0x80 >> idx),
        };

    let masked = byte & mask;
    if masked == 0 {
        Bit::Zero
    }
    else {
        Bit::One
    }
}

pub fn bit_array(byte: u8, bo: BitOrder) -> [Bit; 8] {
    [
        nth(byte, 0, bo),
        nth(byte, 1, bo),
        nth(byte, 2, bo),
        nth(byte, 3, bo),

        nth(byte, 4, bo),
        nth(byte, 5, bo),
        nth(byte, 6, bo),
        nth(byte, 7, bo),
    ]
}

// TODO add a BitOrder argument
pub fn bit_mask(nbits: u8) -> u8 {
    ((1u16 << nbits) - 1) as u8
}

/* trait WriteBit */

pub trait WriteBit {
    fn write(&mut self, b: Bit) -> io::Result<()>;

    fn write_byte(&mut self, n: u8) -> io::Result<()>;

    fn pad_and_flush(&mut self) -> io::Result<()>;
}

/* IoBitWriter */

pub struct IoBitWriter<W> {
    writer:    W,
    curr_bits: u8,
    bit_count: u8,
    bit_order: BitOrder,
}

impl<W> IoBitWriter<W> {
    pub fn new(writer: W, bo: BitOrder) -> Self {
        IoBitWriter {
            writer,
            curr_bits: 0,
            bit_count: 0,
            bit_order: bo
        }
    }

    pub fn into_inner(self) -> W {
        self.writer
    }
}

impl<W: io::Write> WriteBit for IoBitWriter<W> {
    fn write(&mut self, b: Bit) -> io::Result<()> {
        push_bit(&mut self.curr_bits, b, self.bit_order);

        self.bit_count += 1;
        if self.bit_count == 8 {
            self.writer.write_all(&[self.curr_bits])?;
            self.bit_count = 0;
        }

        Ok(())
    }

    fn write_byte(&mut self, n: u8) -> io::Result<()> {
        if self.bit_count == 0 {
            self.writer.write_all(&[n])?;
        }
        else {
            // FIXME this is specific to Msb ordering
            let msb = self.curr_bits << (8 - self.bit_count);
            let lsb = n >> self.bit_count;
            let completed_byte = msb | lsb;
            self.writer.write_all(&[completed_byte])?;

            // Note: this masking is actually not necessary
            self.curr_bits = n & bit_mask(self.bit_count);
        }

        Ok(())
    }

    #[allow(unused_parens)]
    fn pad_and_flush(&mut self) -> io::Result<()> {
        if self.bit_count != 0 {
            // Pad the remains with 0s
            self.curr_bits <<= (8 - self.bit_count);
            self.writer.write_all(&[self.curr_bits])?;

            self.curr_bits = 0;
            self.bit_count = 0;
        }
        self.writer.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_of_b0() {
        assert_eq!(Bit::Zero.byte(), 0u8);
    }

    #[test]
    fn byte_of_b1() {
        assert_eq!(Bit::One.byte(), 1u8);
    }

    fn vec_bw() -> IoBitWriter<Vec<u8>> {
        IoBitWriter::new(Vec::new(), BitOrder::MsbFirst)
    }

    #[test]
    fn write_x00() -> io::Result<()> {
        let mut bw = vec_bw();
        for _ in 0..8 {
            bw.write(Bit::Zero)?;
        }
        assert_eq!(bw.into_inner(), vec![0]);
        Ok(())
    }

    #[test]
    fn write_xff() -> io::Result<()> {
        let mut bw = vec_bw();
        for _ in 0..8 {
            bw.write(Bit::One)?;
        }
        assert_eq!(bw.into_inner(), vec![0xffu8]);
        Ok(())
    }

    #[test]
    fn interleaved_byte() -> io::Result<()> {
        let mut bw = vec_bw();

        // Write 0b010
        bw.write(Bit::Zero)?;
        bw.write(Bit::One)?;
        bw.write(Bit::Zero)?;

        // Write 0xf0
        bw.write_byte(0xf0)?;

        // Write 0b01100
        bw.write(Bit::Zero)?;
        bw.write(Bit::One)?;
        bw.write(Bit::One)?;
        bw.write(Bit::Zero)?;
        bw.write(Bit::Zero)?;

        // Total sequence is 0101 1110 0000 1100
        //                 = 0x5e0c
        assert_eq!(bw.into_inner(), vec![0x5e, 0x0c]);
        Ok(())
    }

    #[test]
    fn bw_mask() -> io::Result<()> {
        let mut bw = vec_bw();

        // Write 0b0000
        bw.write(Bit::Zero)?;
        bw.write(Bit::Zero)?;
        bw.write(Bit::Zero)?;
        bw.write(Bit::Zero)?;

        // Write 0b110
        bw.write(Bit::One)?;
        bw.write(Bit::One)?;
        bw.write(Bit::Zero)?;

        // Write 0xff
        bw.write_byte(0xff)?;

        // Write 0b0
        bw.write(Bit::Zero)?;

        // Total sequence is 0000 1101 1111 1110
        //                 = 0x0dfe
        assert_eq!(bw.into_inner(), vec![0x0d, 0xfe]);
        Ok(())
    }

    #[test]
    fn bit_mask0() {
        assert_eq!(super::bit_mask(0), 0)
    }

    #[test]
    fn bit_mask1() {
        assert_eq!(super::bit_mask(1), 0b1)
    }

    #[test]
    fn bit_mask2() {
        assert_eq!(super::bit_mask(2), 0b11)
    }

    #[test]
    fn bit_mask3() {
        assert_eq!(super::bit_mask(3), 0b111)
    }

    #[test]
    fn bit_mask4() {
        assert_eq!(super::bit_mask(4), 0b1111)
    }

    #[test]
    fn bit_mask5() {
        assert_eq!(super::bit_mask(5), 0b11111)
    }

    #[test]
    fn bit_mask6() {
        assert_eq!(super::bit_mask(6), 0b111111)
    }

    #[test]
    fn bit_mask7() {
        assert_eq!(super::bit_mask(7), 0b1111111)
    }

    #[test]
    fn bit_mask8() {
        assert_eq!(super::bit_mask(8), 0b11111111)
    }

    #[test]
    fn bit_mask9() {
        assert_eq!(super::bit_mask(9), 0b11111111)
    }

    #[test]
    fn nth_lsb() {
        let byte = 0b10110010;
        //           76543210
        assert_eq!(super::nth(byte, 0, BitOrder::LsbFirst), Bit::Zero);
        assert_eq!(super::nth(byte, 1, BitOrder::LsbFirst), Bit::One) ;
        assert_eq!(super::nth(byte, 2, BitOrder::LsbFirst), Bit::Zero);
        assert_eq!(super::nth(byte, 3, BitOrder::LsbFirst), Bit::Zero);

        assert_eq!(super::nth(byte, 4, BitOrder::LsbFirst), Bit::One) ;
        assert_eq!(super::nth(byte, 5, BitOrder::LsbFirst), Bit::One) ;
        assert_eq!(super::nth(byte, 6, BitOrder::LsbFirst), Bit::Zero);
        assert_eq!(super::nth(byte, 7, BitOrder::LsbFirst), Bit::One) ;
    }

    #[test]
    fn nth_msb() {
        let byte = 0b10110010;
        //           01234567
        assert_eq!(super::nth(byte, 0, BitOrder::MsbFirst), Bit::One) ;
        assert_eq!(super::nth(byte, 1, BitOrder::MsbFirst), Bit::Zero);
        assert_eq!(super::nth(byte, 2, BitOrder::MsbFirst), Bit::One) ;
        assert_eq!(super::nth(byte, 3, BitOrder::MsbFirst), Bit::One) ;

        assert_eq!(super::nth(byte, 4, BitOrder::MsbFirst), Bit::Zero);
        assert_eq!(super::nth(byte, 5, BitOrder::MsbFirst), Bit::Zero);
        assert_eq!(super::nth(byte, 6, BitOrder::MsbFirst), Bit::One) ;
        assert_eq!(super::nth(byte, 7, BitOrder::MsbFirst), Bit::Zero);
    }
}
