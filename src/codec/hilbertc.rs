use std::str::FromStr;

use image::{GenericImageView, ImageBuffer, Pixel, Rgb};

use crate::{geom::Distance, hilbert, prs, ser::{Deserialize, Serialize}};

use super::Codec;

pub struct Hilbert {
    compress: CompressionMethod
}

enum CompressionMethod {
    /// Run-Length Encoding.
    RLE(f64),
}

// TODO revert
type RepCount = u8;

impl Codec for Hilbert {
    fn encode<W: std::io::Write>(&self, img: &super::Img, writer: &mut W) -> std::io::Result<()> {
        img.dimensions().serialize(writer)?;

        let iter = hilbert::linearize(img).map(|px| px.to_rgb());

        match &self.compress {
            CompressionMethod::RLE(d)
            if *d == 0.0 => {
                println!("RLE: using exact");
                for (count, color) in rle_exact(iter) {
                    count.serialize(writer)?;
                    color.serialize(writer)?;
                }
            },
            CompressionMethod::RLE(d) => {
                println!("RLE: using approx");
                for (count, color) in rle_approx(iter, *d) {
                    count.serialize(writer)?;
                    color.serialize(writer)?;
                }
            },
        }

        Ok(())
    }

    fn decode<I: Iterator<Item = u8>>(&self, reader: &mut I) -> Option<super::Img> {
        let dimensions = <(u32, u32)>::deserialize(reader)?;
        let mut img_buffer = ImageBuffer::new(dimensions.0, dimensions.1);

        match &self.compress {
            CompressionMethod::RLE(_) => {},
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
            CompressionMethod::RLE(d) if *d == 0.0 => String::from("hilbert-rle"),
            CompressionMethod::RLE(d) => format!("hilbert-rle-approx_{}", d),
        }
    }

    fn is_lossless(&self) -> bool {
        match &self.compress {
            CompressionMethod::RLE(d) => *d == 0.0,
        }
    }
}

/* rle_exact */

fn rle_exact<I: Iterator<Item = T>, T: Eq>(iter: I) -> impl Iterator<Item = (RepCount, T)> {
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

/* rle_approx */

fn rle_approx<I: Iterator<Item = Rgb<u8>>>(iter: I, d: f64) -> impl Iterator<Item = (RepCount, Rgb<u8>)> {
    RleApprox::new(iter, d)
}

struct RleApprox<I> {
    iter:  I,
    last:  Option<Rgb<u8>>,
    allow: f64
}

impl<I> RleApprox<I> {
    fn new(iter: I, allow: f64) -> Self {
        RleApprox {
            iter,
            last: None,
            allow
        }
    }
}

impl<I: Iterator<Item=Rgb<u8>>> Iterator for RleApprox<I> {
    type Item = (RepCount, Rgb<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        let curr_val =
            self.last
                .take()
                .or_else(|| self.iter.next())?;

        let mut running_avg = RunningAvg::new(&curr_val);
        let mut count = 1;
        // TODO can we use a for loop here?
        while let Some(x) = self.iter.next() {
            // TODO introduce an abstract RLE
            if running_avg.avg_f64().dist(&as_f64(&x)) <= self.allow {
                count += 1;
                running_avg.add(&x);
                if count == RepCount::MAX {
                    // We reached the max number of repetitions
                    // Stop early
                    println!("Max number of repetitions reached: {}", count);
                    return Some((count, running_avg.avg()));
                }
            }
            else {
                self.last = Some(x);
                return Some((count, running_avg.avg()));
            }
        }

        // The underlying stream stopped
        // We still need to output our current subsequence
        return Some((count, running_avg.avg()));
    }
}

const CHANNEL_COUNT: usize = 3;

/// A running average for [Rgb<u8>]
struct RunningAvg {
    sum:   [f64; CHANNEL_COUNT],
    count: usize
}

impl RunningAvg {
    fn new(init_color: &Rgb<u8>) -> Self {
        // print!("\n{:?}", init_color);
        RunningAvg {
            sum:   as_f64(init_color),
            count: 1
        }
    }

    fn add(&mut self, x: &Rgb<u8>) {
        // print!(" + {:?}", x);
        for i in 0..CHANNEL_COUNT {
            self.sum[i] += x.0[i] as f64;
        }
        self.count += 1;
    }

    fn avg_f64(&self) -> [f64; CHANNEL_COUNT] {
        self.sum
            .map(|x| x / (self.count as f64))
    }

    fn avg(&self) -> Rgb<u8> {
        let color_f64 = self.avg_f64();
        let safe_convert = |x: f64| {
            assert!(x <= u8::MAX as f64, "{}", x);
            x as u8
        };
        let color_u8 = Rgb(color_f64.map(safe_convert));
        // print!(" = {:?}", &color_u8);
        return color_u8;
    }
}

fn as_f64(color: &Rgb<u8>) -> [f64; CHANNEL_COUNT] {
    color.0
        .map(|x| x as f64)
}

impl Distance for [f64; CHANNEL_COUNT] {
    fn dist(&self, other: &Self) -> f64 {
        let mut dist = 0.0;
        for i in 0..CHANNEL_COUNT {
            dist += (self[i] - other[i]).powi(2);
        }
        dist.sqrt()
    }
}

/* Arg parser */

impl FromStr for Hilbert {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (hilbert, args) = prs::fun_call(s)
                                .ok_or_else(|| format!("Can't parse {:?} as a function", s))?;
        let _ = prs::matches_fully(hilbert, "[Hh]ilbert")
                    .ok_or_else(|| format!("Incorrect name: {}", hilbert))?;

        if args.len() != 1 {
            return Err(format!("Wrong number of arguments for hilbert: expected 1, found {}", args.len()));
        }

        let rle_err =
            if prs::matches_fully(args[0], "rle").is_some() {
                // hilbert(rle)
                0.0
            }
            else {
                // hilbert(rle(<d>))
                let (rle, rle_args) = prs::fun_call(args[0])
                                        .ok_or_else(|| format!("Can't parse {:?} as a function", args[0]))?;

                let _ = prs::matches_fully(rle, "rle")
                            .ok_or_else(|| format!("Argument to Hilbert must be rle, found {:?}", rle))?;
                if rle_args.len() > 1 {
                    return Err(format!("Wrong number of arguments for rle: expected 1, found {}", args.len()));
                }
                assert_ne!(rle_args.len(), 0);

                f64::from_str(rle_args.first().unwrap())
                    .map_err(|e| format!("{:?}", e))?
            };

        Ok(Hilbert { compress: CompressionMethod::RLE(rle_err)})
    }
}
