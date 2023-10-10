mod bench;
mod bit;
mod huf;
mod utils;

use bit::WriteBit;

use image::{GenericImageView, Pixel};
use bytesize::ByteSize;

fn main() {
    // Use the open function to load an image from a Path.
    // `open` returns a `DynamicImage` on success.
    let path = "data/DIV2K_valid_HR/0806.png";
    let img = image::open(path).unwrap();

    // The dimensions method returns the images width and height.
    let dimensions = img.dimensions();
    println!("dimensions {:?}", &dimensions);

    // The color method returns the image's `ColorType`.
    let color_type = img.color();
    let color_size = color_type.bytes_per_pixel();
    println!("{:?} -> {} bytes", &color_type, color_size);

    let uncompressed_size = (color_size as u32) * dimensions.0 * dimensions.1;
    println!("Uncompressed size: {}", ByteSize::b(uncompressed_size as u64));

    let png_size = std::fs::metadata(path).unwrap().len();
    println!("PNG size: {}", ByteSize::b(png_size));

    let compression_rate = 1f64 - ((png_size as f64) / (uncompressed_size as f64));
    println!("PNG compression rate: {:.0}%", compression_rate * 100f64);


    /* Hufman encode */
    let pixels_iter = || img.pixels().map(|(_x, _y, px)| px.to_rgb());
    let freqs = measure(|| utils::count_freqs(pixels_iter()),
                        "Frequency counting");
    let (enc, _dec) = measure(|| huf::build(freqs.into_iter()),
                              "Hufman build");

    let mut bw = bit::IoBitWriter::new(Vec::new());
    measure(|| {
        for px in pixels_iter() {
            enc.encode(&px, &mut bw);
        }
        bw.pad_and_flush();
    }, "Hufman encode");

    let huf_encoded = bw.into_inner();
    let huf_size = huf_encoded.len();
    println!("Hufman size: {}", ByteSize::b(huf_size as u64));

    /* TODO turn into a function */
    let compression_rate = 1f64 - ((huf_size as f64) / (uncompressed_size as f64));
    println!("Hufman (data only) compression rate: {:.0}%", compression_rate * 100f64);
}

fn measure<F: FnOnce() -> T, T>(f: F, description: &str) -> T {
    let start = std::time::Instant::now();
    let res = f();
    let duration = start.elapsed();

    println!("{}: {:?}", description, duration);
    return res;
}
