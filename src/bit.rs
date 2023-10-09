#[derive(Copy, Clone, PartialEq, Eq)]
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
