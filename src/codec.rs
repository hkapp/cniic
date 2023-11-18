mod hufc;
mod clusterc;

use std::io;

pub type Img = image::DynamicImage;

pub trait Codec {
    fn encode<W: io::Write>(&self, img: &Img, writer: &mut W) -> io::Result<()>;
    fn decode<I: Iterator<Item = u8>>(&self, reader: &mut I) -> Option<Img>;
    fn name(&self) -> String;
}

pub use clusterc::RedColKM;
