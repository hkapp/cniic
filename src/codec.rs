mod clusterc;
mod hilbertc;
mod hufc;
mod zipc;

use std::io;
use std::str::FromStr;
use image::{ImageBuffer, Pixel, Rgb};
use crate::prs::{ParseError, ParseAlternatives};
use crate::ser::Deserialize;

pub type Img = image::DynamicImage;

pub trait Codec {
    fn encode<W: io::Write>(&self, img: &Img, writer: &mut W) -> io::Result<()>;
    fn decode<I: Iterator<Item = u8>>(&self, reader: &mut I) -> Option<Img>;
    fn name(&self) -> String;
    fn is_lossless(&self) -> bool;
}

// Utility function for the sub modules
fn create_image_buffer_standard<I: Iterator<Item = u8>>(reader: &mut I) -> Option<((u32, u32), ImageBuffer<Rgb<u8>, Vec<<Rgb<u8> as Pixel>::Subpixel>>)> {
    let dimensions = <(u32, u32)>::deserialize(reader)?;
    let img_buffer = ImageBuffer::new(dimensions.0, dimensions.1);
    Some((dimensions, img_buffer))
}

// To add new codecs, simply add an entry in the gen_all
// macro invocation at the bottom of this file

macro_rules! gen_struct {
    ( $( $mod:ident, $codec:ident );* ) => {
        pub enum AnyCodec {
            $(
                $codec($mod::$codec),
            )*
        }
    };
}

macro_rules! gen_from_str_impl {
    ( $( $mod:ident, $codec:ident );* ) => {
        impl FromStr for AnyCodec {
            type Err = ParseError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                ParseAlternatives::new(s)
                $(
                    .then_try(stringify!($codec), |s| {
                        $mod::$codec::from_str(s)
                            .map(Into::into)
                            .map_err(Into::into)
                    })
                )*
                .end()
            }
        }
    };
}

macro_rules! gen_codec_impl {
    ( $( $codec:ident );* ) => {
        impl Codec for AnyCodec {
            fn encode<W: io::Write>(&self, img: &Img, writer: &mut W) -> io::Result<()> {
                match self {
                    $(
                        AnyCodec::$codec(c) => c.encode(img, writer),
                    )*
                }
            }

            fn decode<I: Iterator<Item = u8>>(&self, reader: &mut I) -> Option<Img> {
                match self {
                    $(
                        AnyCodec::$codec(c) => c.decode(reader),
                    )*
                }
            }

            fn name(&self) -> String {
                match self {
                    $(
                        AnyCodec::$codec(c) => c.name(),
                    )*
                }
            }

            fn is_lossless(&self) -> bool {
                match self {
                    $(
                        AnyCodec::$codec(c) => c.is_lossless(),
                    )*
                }
            }
        }
    };
}

macro_rules! gen_from_impl {
    ( $( $mod:ident, $codec:ident );* ) => {
        $(
            impl From<$mod::$codec> for AnyCodec {
                fn from(h: $mod::$codec) -> Self {
                    AnyCodec::$codec(h)
                }
            }
        )*
    };
}

macro_rules! gen_all {
    ( $( $mod:ident, $codec:ident );* ) => {
        gen_struct!($($mod, $codec);*);
        gen_from_str_impl!($($mod, $codec);*);
        gen_codec_impl!($($codec);*);
        gen_from_impl!($($mod, $codec);*);
    };
}

gen_all!(
    clusterc, ClusterColors;
    clusterc, VoronoiCluster;
    hilbertc, Delta;
    hilbertc, Hilbert;
    hufc, Hufman;
    zipc, Zip
);
