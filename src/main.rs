mod bench;
mod bit;
mod huf;
mod kmeans;
mod utils;
mod ser;

use bit::WriteBit;
use ser::Serialize;

use image::{GenericImageView, Pixel};
use std::io;
use std::collections::{HashSet, HashMap};

fn main() {
    // Bash performs '*' expansion
    let file_paths = std::env::args().skip(1);
    bench::measure_all::<_, _, RedColKM>(file_paths).unwrap();
}

/* bench: Hufman */

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

/* bench: Reduce colors via k-means */

type RedColKM = kmeans::Clusters<Rgb<u8>>;
impl bench::Bench for RedColKM {
    fn encode<W: io::Write>(img: &bench::Img, writer: &mut W) -> io::Result<()> {
        let pixels_iter = || img.pixels().map(|(_x, _y, px)| px.to_rgb());

        // For now: only cluster the unique color. Don't take their frequency into account
        let distinct_colors = HashSet::<_>::from_iter(pixels_iter());

        let (w, h) = img.dimensions();
        let nclusters = w + h;
        let clusters = kmeans::cluster(&mut distinct_colors.into_iter(), nclusters as usize);

        //println!("Resulting clusters:");
        //for i in 0..clusters.len() {
            //print!("{:<3?}  ", clusters[i].centroid.0);
            //if i % 10 == 9 {
                //println!("");
            //}
        //}

        // Convert the clusters into a direct lookup map
        let reduced_colors =
            HashMap::<_, _>::from_iter(
                clusters.into_iter()
                    .flat_map(|(centroid, points_in_cluster)| {
                        // Note: using Rc for the new color is a fake good idea:
                        //       the reference would take more space than the repeated value
                        points_in_cluster.into_iter()
                            .map(move |old_color| (old_color, centroid.clone()))
                    }));

        // Generate a color-reduced image
        let reduced_img = image::ImageBuffer::from_fn(w, h, |x, y| {
                                let original_color = &img.get_pixel(x, y).to_rgb();
                                let new_color = reduced_colors.get(&original_color).unwrap();
                                new_color.clone()
                            });
        // Convert from ImageBuffer to DynamicImage
        let reduced_img = reduced_img.into();

        // Hufman-encode the color-reduced image
        Hufman::encode(&reduced_img, writer)
    }

    fn decode<I: Iterator<Item = u8>>(reader: &mut I) -> Option<bench::Img> {
        Hufman::decode(reader)
    }

    fn name() -> String {
        String::from("red-colors-clusters_w+h")
    }
}

impl kmeans::Point for Rgb<u8> {

    fn dist(&self, other: &Self) -> f64 {
        //self.0[..]
            //.iter()
            //.zip(other.0[..].iter())
            //.map(|(x, y)| (*x as f64 - *y as f64).powi(2))
            //.sum::<f64>()
            //.sqrt()
        let f = |i: usize| {
            let x = self[i];
            let y = other[i];
            let diff = x as i32 - y as i32;
            (diff * diff) as f64
        };
        (f(0) + f(1) + f(2)).sqrt()
    }

    fn mean(points: &[Self]) -> Self {
        fn vector_add(mut x: [u64; 3], y: &Rgb<u8>) -> [u64; 3] {
            for i in 0..3 {
                x[i] += y.0[i] as u64;
            }
            x
        }

        if points.len() == 0 {
            return Rgb(Default::default());
        }

        let sum_vector = points.iter()
                            .fold([0, 0, 0], vector_add);

        Rgb (
            sum_vector.map(|x| (x / points.len() as u64) as u8)
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::kmeans::Point;
    use super::*;

    #[test]
    fn rgb_mean() {
        let p1 = Rgb([0, 0, 0]);
        let p2 = Rgb([2, 2, 2]);
        let expected = Rgb([1, 1, 1]);
        assert_eq!(Point::mean(&[p1, p2][..]), expected);
    }

    #[test]
    fn rgb_dist0() {
        let p = Rgb([0, 10, 20]);
        assert_eq!(p.dist(&p), 0.0);
    }

    #[test]
    fn rgb_dist1() {
        let p1 = Rgb([0, 0, 0]);
        let p2 = Rgb([1, 0, 0]);
        assert_eq!(p1.dist(&p2), 1.0);
    }

    #[test]
    fn rgb_dist2() {
        let p1 = Rgb([0, 0, 0]);
        let p2 = Rgb([1, 1, 0]);
        assert_eq!(p1.dist(&p2), 2.0_f64.sqrt());
    }

    #[test]
    fn rgb_dist3() {
        let p1 = Rgb([0, 0, 0]);
        let p2 = Rgb([1, 1, 1]);
        assert_eq!(p1.dist(&p2), 3.0_f64.sqrt());
    }
}
