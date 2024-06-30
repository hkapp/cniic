mod clusterc;
mod hufc;
mod zipc;

use std::io;
use std::str::FromStr;

pub type Img = image::DynamicImage;

pub trait Codec {
    fn encode<W: io::Write>(&self, img: &Img, writer: &mut W) -> io::Result<()>;
    fn decode<I: Iterator<Item = u8>>(&self, reader: &mut I) -> Option<Img>;
    fn name(&self) -> String;
    fn is_lossless(&self) -> bool;
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

type CodecFromStrErr = Vec<(String, String)>;

macro_rules! gen_from_str_impl {
    ( $( $mod:ident, $codec:ident );* ) => {
        impl FromStr for AnyCodec {
            type Err = CodecFromStrErr;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                fn stack_err(mut prev_err: CodecFromStrErr, name: &str, new_err: String) -> CodecFromStrErr {
                    prev_err.push((String::from(name), new_err));
                    prev_err
                }

                Err(Vec::new())
                $(
                    .or_else(|prev_err: CodecFromStrErr| {
                        $mod::$codec::from_str(s)
                            .map(Into::into)
                            .map_err(|new_err| stack_err(prev_err, stringify!($codec), new_err))
                    })
                )*
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
    hufc, Hufman;
    clusterc, ClusterColors;
    clusterc, VoronoiCluster;
    zipc, Zip
);
