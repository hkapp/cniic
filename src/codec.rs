mod hufc;
mod clusterc;

use std::io;
use std::str::FromStr;

pub type Img = image::DynamicImage;

pub trait Codec {
    fn encode<W: io::Write>(&self, img: &Img, writer: &mut W) -> io::Result<()>;
    fn decode<I: Iterator<Item = u8>>(&self, reader: &mut I) -> Option<Img>;
    fn name(&self) -> String;
}

pub enum AnyCodec {
    Hufman(hufc::Hufman),
    Cluster(clusterc::RedColKM),
}

type CodecFromStrErr = Vec<(String, String)>;

// TODO introduce a macro for generating all of this boiler-plate code
impl FromStr for AnyCodec {
    type Err = CodecFromStrErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        fn stack_err(mut prev_err: CodecFromStrErr, name: &str, new_err: String) -> CodecFromStrErr {
            prev_err.push((String::from(name), new_err));
            prev_err
        }

        Err(Vec::new())
            .or_else(|prev_err: CodecFromStrErr|
                hufc::Hufman::from_str(s)
                    .map(Into::into)
                    .map_err(|new_err| stack_err(prev_err, "Hufman", new_err)))
            .or_else(|prev_err|
                clusterc::RedColKM::from_str(s)
                    .map(Into::into)
                    .map_err(|new_err| stack_err(prev_err, "RedColKM", new_err)))
    }
}

impl Codec for AnyCodec {
    fn encode<W: io::Write>(&self, img: &Img, writer: &mut W) -> io::Result<()> {
        match self {
            AnyCodec::Hufman(h)  => h.encode(img, writer),
            AnyCodec::Cluster(c) => c.encode(img, writer)
        }
    }

    fn decode<I: Iterator<Item = u8>>(&self, reader: &mut I) -> Option<Img> {
        match self {
            AnyCodec::Hufman(h)  => h.decode(reader),
            AnyCodec::Cluster(c) => c.decode(reader)
        }
    }

    fn name(&self) -> String {
        match self {
            AnyCodec::Hufman(h)  => h.name(),
            AnyCodec::Cluster(c) => c.name()
        }
    }
}

impl From<hufc::Hufman> for AnyCodec {
    fn from(h: hufc::Hufman) -> Self {
        AnyCodec::Hufman(h)
    }
}

impl From<clusterc::RedColKM> for AnyCodec {
    fn from(c: clusterc::RedColKM) -> Self {
        AnyCodec::Cluster(c)
    }
}
