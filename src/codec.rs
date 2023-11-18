mod huf;
mod cluster;

use std::io;

pub type Img = image::DynamicImage;

pub trait Codec {
    fn encode<W: io::Write>(img: &Img, writer: &mut W) -> io::Result<()>;
    fn decode<I: Iterator<Item = u8>>(reader: &mut I) -> Option<Img>;
    fn name() -> String;
}

pub use cluster::RedColKM;
