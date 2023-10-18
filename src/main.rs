mod bench;
mod bit;
mod huf;
mod utils;
mod ser;

use bit::WriteBit;
use ser::Serialize;

use image::{GenericImageView, Pixel};
use bytesize::ByteSize;
use std::io;

fn main() {
    // Bash performs '*' expansion
    let file_paths = std::env::args().skip(1);
    bench::measure_all::<_, _, Hufman>(file_paths);
}

fn dummy_test() {
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

use image::Rgb;
type Hufman = (huf::Enc<Rgb<u8>>, huf::Dec<Rgb<u8>>);

static mut saved_dec: Option<huf::Dec<Rgb<u8>>> = None;

impl bench::Bench for Hufman {
    fn encode<W: io::Write>(img: &bench::Img, writer: &mut W) {
        println!("huf_bench::encode: Encoding an image of {}x{}", img.dimensions().0, img.dimensions().1);
        let pixels_iter = || img.pixels().map(|(_x, _y, px)| px.to_rgb());
        let freqs = utils::count_freqs(pixels_iter());
        let (enc, dec) = huf::build(freqs.into_iter());

        // DEBUG encode part of the data and search for a difference
        let mut bw = bit::IoBitWriter::new(Vec::new());
        let mut px_counter = 0;
        for px in pixels_iter().take(300) {
            enc.encode(&px, &mut bw);
            px_counter += 1;
        }
        bw.pad_and_flush();
        let message = bw.into_inner();

        let mut bits = message.into_iter().flat_map(
                            |n| bit::bit_array(n).into_iter().rev());
        for (px_counter, px) in pixels_iter().enumerate() {
            match dec.decode(&mut bits) {
                Some(dec_px) => {
                    if &px != dec_px {
                        println!("Difference at pixel {}: decoded = {:?}, original = {:?}",
                                px_counter, px, dec_px);
                        break;
                    }
                },
                None => {
                    println!("Failed to decode the {}th symbol", px_counter);
                    break;
                }
            }
        }
        // DEBUG end

        // Start by serializing the decoder
        dec.serialize(writer);

        unsafe {
            saved_dec = Some(dec);
        }

        // Write the dimensions of the image
        img.dimensions().serialize(writer);

        // Now write the data
        let mut bw = bit::IoBitWriter::new(writer);
        let mut px_counter = 0;
        for px in pixels_iter() {
            if px_counter == 296 {
                println!("Px input: {:?}", &px);
            }
            else {
                //panic!("That should be enough");
            }
            enc.encode(&px, &mut bw);
            px_counter += 1;
        }
        bw.pad_and_flush();
        bw.into_inner().flush();
    }

    fn decode<I: Iterator<Item = u8>>(reader: &mut I) -> Option<bench::Img> {
        use ser::Deserialize;

        println!("Start decoding...");

        // Start by reading the decoder
        let dec: huf::Dec<image::Rgb<u8>> = Deserialize::deserialize(reader)?;
        println!("Deserialized the decoder");
        println!("It {} match the serialized one",
                    if Some(&dec) == unsafe { saved_dec.as_ref() }
                        { "DOES" }
                    else
                        { "does NOT" });

        // Read the dimensions of the image
        // TODO make the dims into a type
        let dims: (u32, u32) = Deserialize::deserialize(reader)?;
        println!("Deserialized the image dimensions: {}x{}", dims.0, dims.1);

        // Read the data and create the image
        // FIXME introduce MsbFirst / LsbFirst
        let mut bits = reader.flat_map(
                            |n| bit::bit_array(n).into_iter().rev());
        let mut img = image::RgbImage::new(dims.0, dims.1);
        let mut px_counter = 0;
        for px in img.pixels_mut() {
            //*px = *dec.decode(&mut bits)?;
            match dec.decode(&mut bits) {
                Some(x) => {
                    *px = *x;
                    if px_counter == 296 {
                        println!("{:?}", *x);
                    }
                    else {
                        //panic!("You've seen enough");
                    }
                    px_counter += 1;
                },
                None => {
                    println!("Failed to decode the {}th symbol", px_counter);
                    if reader.next().is_none() {
                        println!("Reached the end of the stream");
                    }
                    return None;
                }
            }
        }
        println!("Read all the pixels");

        Some(img.into())
    }

    fn name() -> String {
        String::from("Hufman")
    }
}
