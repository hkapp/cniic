use zhang_hilbert::ArbHilbertScan32;
use image::GenericImageView;

pub fn linearize<T: GenericImageView>(img: &T) -> impl Iterator<Item = T::Pixel> + '_ {
    from_matrix(img)
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
