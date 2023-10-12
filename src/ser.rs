use std::io;

pub trait Serialize {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()>;
}

pub trait Deserialize {
    fn deserialize<R: io::Read>(&self, reader: &mut R) -> io::Result<()>;
}

/* Standard implementations */

/* u8 */

impl Serialize for u8 {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&[*self])
    }
}

/* u32 */

impl Serialize for u32 {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&self.to_le_bytes())
    }
}

/* u32 */

impl Serialize for usize {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&self.to_le_bytes())
    }
}

/* Tuple2 */

impl<T: Serialize, U: Serialize> Serialize for (T, U) {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        self.0.serialize(writer)?;
        self.1.serialize(writer)
    }
}

/* &[] / Vec */

impl<T: Serialize> Serialize for &[T] {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        self.len().serialize(writer)?;
        for x in self.iter() {
            x.serialize(writer)?;
        }
        Ok(())
    }
}

/* image::Rgb */
/* Arguable: should be somewhere else */

impl<T: Serialize> Serialize for image::Rgb<T> {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        (&self.0[..]).serialize(writer)
    }
}
