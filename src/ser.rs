use std::{collections::VecDeque, io, iter, slice};

pub trait Serialize {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()>;
}

pub trait Deserialize
    where Self: Sized
{
    fn deserialize<I: Iterator<Item = u8>>(stream: &mut I) -> Option<Self>;
}

/* Standard implementations */

/* u8 */

impl Serialize for u8 {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(slice::from_ref(self))
    }
}

impl Deserialize for u8 {
    fn deserialize<I: Iterator<Item = u8>>(stream: &mut I) -> Option<Self> {
        stream.next()
    }
}

/* u16 */

impl Serialize for u16 {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&self.to_le_bytes())
    }
}

impl Deserialize for u16 {
    fn deserialize<I: Iterator<Item = u8>>(stream: &mut I) -> Option<Self> {
        let n = u16::from_le_bytes([
            stream.next()?,
            stream.next()?,
        ]);
        Some(n)
    }
}

/* u32 */

impl Serialize for u32 {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&self.to_le_bytes())
    }
}

impl Deserialize for u32 {
    fn deserialize<I: Iterator<Item = u8>>(stream: &mut I) -> Option<Self> {
        let n = u32::from_le_bytes([
            stream.next()?,
            stream.next()?,
            stream.next()?,
            stream.next()?,
        ]);
        Some(n)
    }
}

/* u64 */

impl Serialize for u64 {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&self.to_le_bytes())
    }
}

impl Deserialize for u64 {
    fn deserialize<I: Iterator<Item = u8>>(stream: &mut I) -> Option<Self> {
        let n = u64::from_le_bytes([
            stream.next()?,
            stream.next()?,
            stream.next()?,
            stream.next()?,

            stream.next()?,
            stream.next()?,
            stream.next()?,
            stream.next()?,
        ]);
        Some(n)
    }
}

/* usize */
/* Normalized to u64 */

impl Serialize for usize {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        (*self as u64).serialize(writer)
    }
}

impl Deserialize for usize {
    fn deserialize<I: Iterator<Item = u8>>(stream: &mut I) -> Option<Self> {
        let n: u64 = Deserialize::deserialize(stream)?;
        Some(n as usize)
    }
}

/* char */
/* FIXME we only support ascii atm */

impl Serialize for char {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        let ascii: u8 = u8::try_from(*self)
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        ascii.serialize(writer)
    }
}

impl Deserialize for char {
    fn deserialize<I: Iterator<Item = u8>>(stream: &mut I) -> Option<Self> {
        let ascii: u8 = Deserialize::deserialize(stream)?;
        Some(ascii.into())
    }
}

/* Tuple2 */

impl<T: Serialize, U: Serialize> Serialize for (T, U) {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        self.0.serialize(writer)?;
        self.1.serialize(writer)
    }
}

impl<T: Deserialize, U: Deserialize> Deserialize for (T, U) {
    fn deserialize<I: Iterator<Item = u8>>(stream: &mut I) -> Option<Self> {
        let a = Deserialize::deserialize(stream)?;
        let b = Deserialize::deserialize(stream)?;
        Some((a, b))
    }
}

/* &[] / Vec */
/* Note: these two work together */

impl<T: Serialize> Serialize for &[T] {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        self.len().serialize(writer)?;
        for x in self.iter() {
            x.serialize(writer)?;
        }
        Ok(())
    }
}

impl<T:Deserialize> Deserialize for Vec<T> {
    fn deserialize<I: Iterator<Item = u8>>(stream: &mut I) -> Option<Self> {
        let len: usize = Deserialize::deserialize(stream)?;

        let mut vec = Vec::new();
        for _ in 0..len {
            vec.push(Deserialize::deserialize(stream)?);
        }
        Some(vec)
    }
}

/* image::Rgb */
/* Arguable: should be somewhere else */

impl<T: Serialize> Serialize for image::Rgb<T> {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        (&self.0[..]).serialize(writer)
    }
}

impl<T:Deserialize> Deserialize for image::Rgb<T> {
    fn deserialize<I: Iterator<Item = u8>>(stream: &mut I) -> Option<Self> {
        let vec: Vec<T> = Deserialize::deserialize(stream)?;
        let arr: [T; 3] = vec.try_into().ok()?;
        Some(arr.into())
    }
}

/* Utility: SerStream */

pub struct SerStream<I> {
    iter:   I,
    extra:  VecDeque<u8>,
}

impl<T: Serialize> SerStream<iter::Once<T>> {
    pub fn from_value(value: T) -> Self {
        Self::from_iter(iter::once(value))
    }
}

impl<T: Serialize, I: Iterator<Item=T>> SerStream<I> {
    pub fn from_iter(iter: I) -> Self {
        SerStream {
            iter,
            extra: VecDeque::new()
        }
    }
}

impl<T: Serialize, I: Iterator<Item = T>> Iterator for SerStream<I> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        while self.extra.is_empty() {
            match self.iter.next() {
                Some(x) => {
                    x.serialize(&mut self.extra);
                }
                None => {
                    // No more elements, we're done forever
                    return None;
                }
            }
        }
        self.extra
            .pop_front()
    }
}

// pub struct SerStream {
//     iters: VecDeque<Box<dyn Iterator<Item=u8>>>,
//     curr_iter: Option<Box<dyn Iterator<Item=u8>>>,
// }

// impl SerStream {
//     pub fn new() -> Self {
//         SerStream {
//             iters:     VecDeque::new(),
//             curr_iter: None,
//         }
//     }

//     pub fn push_iter<I: Iterator<Item=T>, T: Serialize>(&mut self, new_iter: I) {
//         let byte_iter = new_iter.flat_map(|x| )
//         self.iters
//             .push_back()
//     }
// }
