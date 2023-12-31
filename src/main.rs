mod bench;
mod bit;
mod huf;
mod utils;
mod ser;

use bit::WriteBit;
use ser::Serialize;

use image::{GenericImageView, Pixel};
use std::io;

fn main() {
    // Bash performs '*' expansion
    let file_paths = std::env::args().skip(1);
    bench::measure_all::<_, _, Hufman>(file_paths).unwrap();
}

use image::Rgb;
type Hufman = (huf::Enc<Rgb<u8>>, huf::Dec<Rgb<u8>>);

impl bench::Bench for Hufman {
    fn encode<W: io::Write>(img: &bench::Img, writer: &mut W) -> io::Result<()> {
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

    fn decode<I: Iterator<Item = u8>>(reader: &mut I) -> Option<bench::Img> {
        use ser::Deserialize;

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

    fn name() -> String {
        String::from("Hufman")
    }
}
