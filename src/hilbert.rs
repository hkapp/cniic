use std::cmp::{max, min};

use zhang_hilbert::ArbHilbertScan32;
use image::GenericImageView;

pub use linearize_rect as linearize;

/// Linearize an image using zhang's arbitrary rectangle's pseudo-Hilbert curve.
/// This is the default.
pub fn linearize_rect<T: GenericImageView>(img: &T) -> impl Iterator<Item = T::Pixel> + '_ {
    from_matrix(img)
}

/// Linearize a picture into the largest square that is totally enscribed in it.
pub fn linearize_small<T: GenericImageView>(img: &T) -> impl Iterator<Item = T::Pixel> + '_ {
    let (xdim, ydim) = img.dimensions();

    let square_order = min(xdim.next_power_of_two() >> 1, ydim.next_power_of_two() >> 1) as usize;

    iter(square_order, square_order)
        .map(|(x, y)| img.get(x, y))
}

/// Linearize a picture using the smallest square that contains it.
pub fn linearize_large<T: GenericImageView>(img: &T) -> impl Iterator<Item = T::Pixel> + '_ {
    let (xorig, yorig) = img.dimensions();
    let square_order = max(xorig.next_power_of_two(), yorig.next_power_of_two());

    iter(square_order as usize, square_order as usize)
        .filter(move |(x, y)| *x < xorig as usize && *y < yorig as usize)
        .map(|(x, y)| img.get(x, y))
}

fn from_matrix<T: Matrix>(data: &T) -> impl Iterator<Item = T::Item> + '_ {
    let (xdim, ydim) = data.dimensions();
    iter(xdim, ydim)
        .map(|(x, y)| data.get(x, y))
}

fn iter(xdim: usize, ydim: usize) -> impl Iterator<Item = (usize, usize)> {
    ArbHilbertScan32::new([xdim.try_into().unwrap(), ydim.try_into().unwrap()])
        .map(|arr| (arr[0] as usize, arr[1] as usize))
}

trait Matrix {
    type Item;

    fn dimensions(&self) -> (usize, usize);
    fn get(&self, x: usize, y: usize) -> Self::Item;
}

impl<T: GenericImageView> Matrix for T {
    type Item = T::Pixel;

    fn dimensions(&self) -> (usize, usize) {
        let dim32 = self.dimensions();
        (dim32.0.try_into().unwrap(), dim32.1.try_into().unwrap())
    }

    fn get(&self, x: usize, y: usize) -> Self::Item {
        self.get_pixel(x.try_into().unwrap(), y.try_into().unwrap())
    }
}
