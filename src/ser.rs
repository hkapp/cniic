use std::io;

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
        writer.write_all(&[*self])
    }
}

impl Deserialize for u8 {
    fn deserialize<I: Iterator<Item = u8>>(stream: &mut I) -> Option<Self> {
        stream.next()
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
