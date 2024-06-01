use crate::kmeans;
use image::{GenericImageView, Pixel, Rgb};
use std::io;
use std::collections::HashMap;
use super::hufc::Hufman;
use super::{Codec, Img};
use std::str::FromStr;
use regex::Regex;
use crate::geom::Distance;
use crate::utils;

/* codec: Reduce colors via k-means, then apply Hufman */

pub struct ClusterColors(usize);  // number of colors to reduce to

impl Codec for ClusterColors {
    fn encode<W: io::Write>(&self, img: &Img, writer: &mut W) -> io::Result<()> {
        let pixels_iter = || img.pixels().map(|(_x, _y, px)| px.to_rgb());

        let color_counts = utils::count_freqs(pixels_iter())
                                .into_iter()
                                .map(|(color, count)| ColorCount { color, count: count as u32 })
                                .collect();

        let (w, h) = img.dimensions();
        let nclusters = self.0;
        let clusters = kmeans::cluster(color_counts, nclusters as usize);

        // Convert the clusters into a direct lookup map
        let reduced_colors =
            HashMap::<_, _>::from_iter(
                clusters.into_iter()
                    .flat_map(|(centroid, points_in_cluster)| {
                        // Note: using Rc for the new color is a fake good idea:
                        //       the reference would take more space than the repeated value
                        points_in_cluster.into_iter()
                            .map(|color_count| color_count.color)
                            .map(move |old_color| (old_color, centroid.color.clone()))
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
        format!("red-colors-clusters_{}", self.0)
    }

    fn is_lossless(&self) -> bool {
        false
    }
}

#[derive(Clone)]
struct ColorCount {
    color: Rgb<u8>,
    count: u32  // what GenericImageView would return
}

impl Distance for ColorCount {
    // We only compute the distance between the points, ignoring the weights
    fn dist(&self, other: &Self) -> f64 {
        self.color.dist(&other.color)
    }
}

impl kmeans::Point for ColorCount {
    // Weighted average of the colors
    fn mean(points: &[Self]) -> Option<Self> {
        if points.len() == 0 {
            return None;
        }
        else if points.len() == 1 {
            // Behave like 'clone()'
            return Some(points[0].clone());
        }

        fn weighed_add(mut x: ([u64; 3], u64), y: &ColorCount) -> ([u64; 3], u64) {
            for i in 0..3 {
                x.0[i] += (y.color[i] as u64) * (y.count as u64);
            }
            x.1 += y.count as u64;
            x
        }
        let (sum_vector, tot_weight) = points.iter()
                                            .fold(([0, 0, 0], 0), weighed_add);

        let avg_color =
            Rgb (
                sum_vector.map(|x| (x / tot_weight) as u8)
            );

        Some(
            ColorCount {
                color: avg_color,
                count: 1
            }
        )
    }
}

impl FromStr for ClusterColors {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        /* Allow:
         *   redcol(x)
         *   reduce-colors(x)
         * and all the in-between
         */
        let regexp = Regex::new(r"red(?:uce)?-?col(?:ors)?\((\d+)\)").unwrap();
        let matches = regexp.captures(s)
                            .ok_or(String::from("Regex doesn't match"))?;

        // We expect one capture group for the entire match,
        // and one for out parenthesized expression.
        if matches.len() != 2 {
            return Err("Couldn't parse the numeric argument".into());
        }

        let digits_str = matches.get(1).unwrap();
        let ncolors = usize::from_str(digits_str.as_str())
                            .map_err(|e| format!("{:?}", e))?;

        Ok(ClusterColors(ncolors))
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
