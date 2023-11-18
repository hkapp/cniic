use crate::{huf, utils};
use image::{GenericImageView, Pixel};
use std::io;
use crate::ser::{Serialize, Deserialize};
use crate::bit::{self, WriteBit};
use super::{Codec, Img};
use std::str::FromStr;

/* codec: Hufman */

pub struct Hufman;

impl Codec for Hufman {
    fn encode<W: io::Write>(&self, img: &Img, writer: &mut W) -> io::Result<()> {
        let pixels_iter = || img.pixels().map(|(_x, _y, px)| px.to_rgb());
        let freqs = utils::count_freqs(pixels_iter());
        let (enc, dec) = huf::build(freqs.into_iter());

        // Start by serializing the decoder
        dec.serialize(writer)?;

        // Write the dimensions of the image
        img.dimensions().serialize(writer)?;

        // Now write the data
        let mut bw = bit::IoBitWriter::new(writer, huf::BIT_ORDER);
        for px in pixels_iter() {
            enc.encode(&px, &mut bw);
        }
        bw.pad_and_flush()?;
        bw.into_inner().flush()?;

        Ok(())
    }

    fn decode<I: Iterator<Item = u8>>(&self, reader: &mut I) -> Option<Img> {
        // Start by reading the decoder
        let dec: huf::Dec<image::Rgb<u8>> = Deserialize::deserialize(reader)?;

        // Read the dimensions of the image
        let dims: (u32, u32) = Deserialize::deserialize(reader)?;

        // Read the data and create the image
        let mut bits = reader.flat_map(
                            |n| bit::bit_array(n, huf::BIT_ORDER).into_iter());
        let mut img = image::RgbImage::new(dims.0, dims.1);
        for px in img.pixels_mut() {
            match dec.decode(&mut bits) {
                Some(x) => {
                    *px = *x;
                },
                None => {
                    eprintln!("Failed to decode symbol");
                    if reader.next().is_none() {
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
