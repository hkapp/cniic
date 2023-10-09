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
            Zero => 0,
            One  => 1,
        }
    }
}

pub fn byte_from_slice(bits: &[Bit]) -> Option<u8> {
    if bits.len() != 8 {
        return None
    }

    let mut n = 0u8;
    for b in bits {
        n <<= 1;
        n &= b.byte();
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
}
