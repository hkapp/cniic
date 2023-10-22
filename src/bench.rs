use std::fs;
use std::io;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use image::GenericImageView;
use bytesize::ByteSize;

pub type Img = image::DynamicImage;

pub trait Bench {
    fn encode<W: io::Write>(img: &Img, writer: &mut W) -> io::Result<()>;
    fn decode<I: Iterator<Item = u8>>(reader: &mut I) -> Option<Img>;
    fn name() -> String;
}

pub fn measure_all<I, P, T>(paths: I) -> io::Result<()>
    where I: Iterator<Item = P>,
        P: AsRef<Path>,
        T: Bench
{
    let mut wrote_header = false;
    let mut csv = csv_writer::<T>()?;

    paths
        .map(|p| {
            let img = image::open(p.as_ref())
                        .map_err(|e| format!("{:?}", e))?;

            let mut data = Vec::new();
            T::encode(&img, &mut data)
                .map_err(|e| format!("{:?}", e))?;

            let compressed_size = data.len();
            println!("bench: Compressed size = {}", ByteSize::b(compressed_size as u64));

            let (h, w) = img.dimensions();
            let raw_size = h * w * 24; // we're counting on 3 bytes RGB pixels
            println!("bench: Raw size = {}", ByteSize::b(raw_size as u64));
            let compression_ratio = (compressed_size as f64) / (raw_size as f64);

            let decoded = T::decode(&mut data.into_iter());
            match decoded {
                Some(test) => {
                    if test != img {
                        /* DEBUG begin
                        let mut px_pairs = test.pixels().zip(img.pixels());
                        let first_difference = px_pairs.find(|(enc_px, exp_px)| enc_px != exp_px);
                        eprintln!("First difference found: {:?}", first_difference);
                        eprintln!("Follow {} differences", px_pairs.count());
                         * DEBUG end
                         */
                        // Write the incorrect image to file
                        let mut path = PathBuf::from("output");
                        path.push(p.as_ref().file_name().unwrap());
                        path.set_extension("png");
                        test.save(&path)
                            .map_err(|e| format!("{:?}", e))?;
                        eprintln!("Saved to {}", path.display());
                        return Err(String::from("Decoded image doesn't match"))
                    }
                }
                None => {
                    return Err(String::from("Could not decode the image"))
                }
            }

            if !wrote_header {
                csv.write_record(&["name", "compressed_size", "compression_ratio"])
                    .map_err(|e| format!("{:?}", e))?;
                wrote_header = true;
            }

            let file_name = p.as_ref().to_str().unwrap_or("???");
            csv.serialize((file_name, compressed_size, compression_ratio * 100.0))
                .map_err(|e| format!("{:?}", e))?;
            Ok(())
        })
        .for_each(|r| print_err(r));

    csv.flush()?;
    Ok(())
}

fn csv_writer<T: Bench>() -> Result<csv::Writer<fs::File>, csv::Error> {
    let mut csv_filename = String::from("output/");
    csv_filename.push_str(&T::name());
    csv_filename.push_str(".csv");

    csv::Writer::from_path(&csv_filename)
}

fn print_err<T, E: Debug>(r: Result<T, E>) {
    match r {
        Err(e) => eprintln!("{:?}", e),
        _      => (),
    }
}

#[allow(dead_code)]
fn measure<F: FnOnce() -> T, T>(f: F, description: &str) -> T {
    let start = std::time::Instant::now();
    let res = f();
    let duration = start.elapsed();

    println!("{}: {:?}", description, duration);
    return res;
}
