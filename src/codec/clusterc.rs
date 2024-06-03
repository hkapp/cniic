use crate::kmeans;
use crate::ser::{Serialize, Deserialize};
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

/* Codec: apply 5d clustering to the pixel, then redraw as a Voronoi diagram */

pub struct VoronoiCluster(usize);  // number of clusters

impl Codec for VoronoiCluster {
    fn encode<W: io::Write>(&self, img: &Img, writer: &mut W) -> io::Result<()> {
        // Cluster the 5d points
        let data_points = img.pixels()
                            .map(|(x, y, c)| ColorPos { x, y, color: c.to_rgb() })
                            .collect();
        let clusters = kmeans::cluster(data_points, self.0);

        // Write the image dimensions
        let (w, h) = img.dimensions();
        w.serialize(writer)?;
        h.serialize(writer)?;

        // Write the centroids
        self.0.serialize(writer)?;
        for (centroid, _points) in clusters.into_iter() {
            centroid.serialize(writer)?;
        }
        Ok(())
    }

    fn decode<I: Iterator<Item = u8>>(&self, reader: &mut I) -> Option<Img> {
        // Read the dimensions of the image
        let dims: (u32, u32) = Deserialize::deserialize(reader)?;
        let mut img = image::RgbImage::new(dims.0, dims.1);

        // Read the centroids
        let ncentroids: usize = Deserialize::deserialize(reader)?;
        let centroids: Vec<_> = (0..ncentroids)
                                    .map(|_| ColorPos::deserialize(reader))
                                    .collect::<Option<Vec<_>>>()?;

        // Re-create the image
        for (x, y, px) in img.enumerate_pixels_mut() {
            // Find the closest centroid
            let closest_centroid = centroids.iter()
                                        .min_by_key(|c| (c.x - x).pow(2) + (c.y - y).pow(2))
                                        .unwrap();
            *px = closest_centroid.color.clone();
        }

        Some(img.into())
    }

    fn name(&self) -> String {
        format!("voronoi_{}", self.0)
    }

    fn is_lossless(&self) -> bool {
        false
    }
}

struct ColorPos {
    x:     u32,
    y:     u32,
    color: Rgb<u8>,
}

impl Distance for ColorPos {
    fn dist(&self, other: &Self) -> f64 {
        let mut d = (self.x - other.x).pow(2) as f64;
        d += (self.y - other.y).pow(2) as f64;
        d += self.color.dist(&other.color).powi(2);
        (d as f64).sqrt()
    }
}

impl kmeans::Point for ColorPos {
    fn mean(points: &[Self]) -> Option<Self> {
        if points.len() == 0 {
            return None;
        }

        fn vector_add(mut x: [u64; 5], y: &ColorPos) -> [u64; 5] {
            x[0] += y.x as u64;
            x[1] += y.y as u64;
            for i in 0..3 {
                x[i+2] += y.color[i] as u64;
            }
            x
        }
        let sum_vector = points.iter()
                                            .fold([0; 5], vector_add);
        let avg_vector = sum_vector.map(|x| x / points.len() as u64);

        Some(
            ColorPos {
                x: avg_vector[0] as u32,
                y: avg_vector[1] as u32,
                color:
                    Rgb (
                        [
                            avg_vector[2] as u8,
                            avg_vector[3] as u8,
                            avg_vector[4] as u8,
                        ]
                    )
            }
        )
    }
}

impl Serialize for ColorPos {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        self.x.serialize(writer)?;
        self.y.serialize(writer)?;
        self.color.serialize(writer)?;
        Ok(())
    }
}

impl Deserialize for ColorPos {
    fn deserialize<I: Iterator<Item = u8>>(stream: &mut I) -> Option<Self> {
        let x: u32 = Deserialize::deserialize(stream)?;
        let y: u32 = Deserialize::deserialize(stream)?;
        let color: Rgb<u8> = Deserialize::deserialize(stream)?;
        Some(
            ColorPos {
                x,
                y,
                color
            }
        )
    }
}

impl FromStr for VoronoiCluster {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        /* Allow:
         *   voronoi(x)
         */
        let regexp = Regex::new(r"voronoi\((\d+)\)").unwrap();
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

        Ok(VoronoiCluster(ncolors))
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
