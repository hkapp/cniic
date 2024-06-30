use std::str::FromStr;

use image::{GenericImageView, ImageBuffer, Pixel, Rgb};

use crate::{hilbert, prs, ser::{Serialize, Deserialize}};

use super::Codec;

pub struct Hilbert {
    compress: CompressionMethod
}

enum CompressionMethod {
    /// Run-Length Encoding.
    RLE,
}

type RepCount = u8;

impl Codec for Hilbert {
    fn encode<W: std::io::Write>(&self, img: &super::Img, writer: &mut W) -> std::io::Result<()> {
        img.dimensions().serialize(writer)?;

        match &self.compress {
            CompressionMethod::RLE => {},
        }

        let iter = hilbert::linearize(img).map(|px| px.to_rgb());
        for (count, color) in rle(iter) {
            count.serialize(writer)?;
            color.serialize(writer)?;
        }
        Ok(())
    }

    fn decode<I: Iterator<Item = u8>>(&self, reader: &mut I) -> Option<super::Img> {
        let dimensions = <(u32, u32)>::deserialize(reader)?;
        let mut img_buffer = ImageBuffer::new(dimensions.0, dimensions.1);

        match &self.compress {
            CompressionMethod::RLE => {},
        }

        let mut count = RepCount::deserialize(reader)?;
        let mut curr_color = <Rgb<u8>>::deserialize(reader)?;
        for (x, y) in hilbert::iter(dimensions.0 as usize, dimensions.1 as usize) {
            if count == 0 {
                count = RepCount::deserialize(reader)?;
                curr_color = <Rgb<u8>>::deserialize(reader)?;
            }

            *img_buffer.get_pixel_mut(x as u32, y as u32) = curr_color;

            count -= 1;
        }

        Some(img_buffer.into())
    }

    fn name(&self) -> String {
        match &self.compress {
            CompressionMethod::RLE => String::from("hilbert-rle"),
        }
    }

    fn is_lossless(&self) -> bool {
        match &self.compress {
            CompressionMethod::RLE => true,
        }
    }
}

fn rle<I: Iterator<Item = T>, T: Eq>(iter: I) -> impl Iterator<Item = (RepCount, T)> {
    Rle::new(iter)
}

struct Rle<I, T> {
    iter: I,
    last: Option<T>
}

impl<I, T> Rle<I, T> {
    fn new(iter: I) -> Self {
        Rle {
            iter,
            last: None
        }
    }
}

impl<I: Iterator<Item=T>, T: Eq> Iterator for Rle<I, T> {
    type Item = (RepCount, T);

    fn next(&mut self) -> Option<Self::Item> {
        let curr_val =
            self.last
                .take()
                .or_else(|| self.iter.next())?;

        let mut count = 1;
        // TODO can we use a for loop here?
        while let Some(x) = self.iter.next() {
            if x == curr_val {
                count += 1;
                if count == RepCount::MAX {
                    // We reached the max number of repetitions
                    // Stop early
                    println!("Max number of repetitions reached: {}", count);
                    return Some((count, curr_val));
                }
            }
            else {
                self.last = Some(x);
                return Some((count, curr_val));
            }
        }

        // The underlying stream stopped
        // We still need to output our current subsequence
        return Some((count, curr_val));
    }
}

impl FromStr for Hilbert {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (hilbert, args) = prs::fun_call(s)
                                .ok_or_else(|| String::from("Incorrect syntax"))?;
        let _ = prs::matches_fully(hilbert, "[Hh]ilbert")
                    .ok_or_else(|| format!("Incorrect name: {}", hilbert))?;

        if args.len() != 1 {
            return Err(format!("Wrong number of arguments: expected 1, found {}", args.len()));
        }

        let compress =
            match args[0] {
                "rle"     => CompressionMethod::RLE,
                a@_ => return Err(format!("Unrecognized argument: {}", a)),
            };
        Ok(Hilbert { compress })
    }
}
