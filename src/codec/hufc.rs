use crate::{huf, ser::Serialize};
use image::{GenericImageView, Pixel, Rgb};
use std::io;
use super::{Codec, Img};
use std::str::FromStr;

/* codec: Hufman */

pub struct Hufman;

impl Codec for Hufman {
    fn encode<W: io::Write>(&self, img: &Img, writer: &mut W) -> io::Result<()> {
        img.dimensions().serialize(writer)?;

        let pixels_iter = || img.pixels().map(|(_x, _y, px)| px.to_rgb());
        huf::encode_all(pixels_iter, writer)
    }

    fn decode<I: Iterator<Item = u8>>(&self, reader: &mut I) -> Option<Img> {
        let (_, mut img) = super::create_image_buffer_standard(reader)?;

        // Read the data and create the image
        let mut symbols = huf::decode_all::<_, Rgb<u8>>(reader)?;
        for px in img.pixels_mut() {
            match symbols.next() {
                Some(x) => {
                    *px = x;
                },
                None => {
                    eprintln!("Failed to decode symbol");
                    if symbols.next().is_none() {
                        eprintln!("Reached the end of the stream");
                    }
                    return None;
                }
            }
        }

        Some(img.into())
    }

    fn name(&self) -> String {
        String::from("Hufman")
    }

    fn is_lossless(&self) -> bool {
        true
    }
}

impl FromStr for Hufman {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // TODO use a regex to be a bit more lenient
        if s.eq_ignore_ascii_case("hufman") {
            Ok(Hufman)
        }
        else {
            Err(String::from("Not Hufman"))
        }
    }
}
