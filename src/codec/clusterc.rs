use crate::kmeans;
use image::{GenericImageView, Pixel, Rgb};
use std::io;
use std::collections::{HashSet, HashMap};
use super::hufc::Hufman;
use super::{Codec, Img};

/* codec: Reduce colors via k-means */

pub struct RedColKM(pub usize);  // number of colors to reduce to

impl Codec for RedColKM {
    fn encode<W: io::Write>(&self, img: &Img, writer: &mut W) -> io::Result<()> {
        let pixels_iter = || img.pixels().map(|(_x, _y, px)| px.to_rgb());

        // For now: only cluster the unique color. Don't take their frequency into account
        let colors_set = HashSet::<_>::from_iter(pixels_iter());
        let distinct_colors = Vec::from_iter(colors_set.into_iter());

        let (w, h) = img.dimensions();
        let nclusters = self.0;
        let clusters = kmeans::cluster(distinct_colors, nclusters as usize);

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
        Hufman.encode(&reduced_img, writer)
    }

    fn decode<I: Iterator<Item = u8>>(&self, reader: &mut I) -> Option<Img> {
        Hufman.decode(reader)
    }

    fn name(&self) -> String {
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

    fn mean(points: &[Self]) -> Option<Self> {
        fn vector_add(mut x: [u64; 3], y: &Rgb<u8>) -> [u64; 3] {
            for i in 0..3 {
                x[i] += y.0[i] as u64;
            }
            x
        }

        if points.len() == 0 {
            return None;
        }

        let sum_vector = points.iter()
                            .fold([0, 0, 0], vector_add);

        Some(
            Rgb (
                sum_vector.map(|x| (x / points.len() as u64) as u8)
            )
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
        assert_eq!(Point::mean(&[p1, p2][..]), Some(expected));
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
