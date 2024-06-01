use std::fs;
use std::io;
use std::fmt::Debug;
use std::path::PathBuf;
use image::GenericImageView;
use bytesize::ByteSize;
use rayon::prelude::*;
use std::sync::RwLock;
use crate::codec::{Codec, Img};
use std::ops::DerefMut;
use std::marker::Sync;
use crate::kmeans::Point;
use image::Pixel;

pub fn measure_all<I, P, T>(codec: &T, paths: I) -> io::Result<()>
    where I: Iterator<Item = P>,
        P: Into<PathBuf>,
        T: Codec + Sync
{
    let wrote_header = false;
    let csv = csv_writer(codec)?;
    let writer_state = RwLock::new((wrote_header, csv));

    paths
        .map(|p| p.into())
        .collect::<Vec<PathBuf>>()
        .into_par_iter()
        .map(|p| {
            println!("Processing {:?}...", p);
            let img = image::open(&p)
                        .map_err(|e| format!("{:?}", e))?;

            let mut data = Vec::new();
            codec.encode(&img, &mut data)
                .map_err(|e| format!("{:?}", e))?;

            let compressed_size = data.len();
            println!("bench: Compressed size = {}", ByteSize::b(compressed_size as u64));

            let (h, w) = img.dimensions();
            let raw_size = h * w * 24; // we're counting on 3 bytes RGB pixels
            println!("bench: Raw size = {}", ByteSize::b(raw_size as u64));
            let compression_ratio = (compressed_size as f64) / (raw_size as f64);

            let decoded = codec.decode(&mut data.into_iter())
                                .ok_or_else(|| String::from("Could not decode the image"))?;

            let error = compute_error(&img, &decoded);

            if error != 0.0 {
                // Write the incorrect image to file
                let mut path = PathBuf::from("output");
                path.push(p.file_name().unwrap());
                path.set_extension("png");
                decoded.save(&path)
                        .map_err(|e| format!("{:?}", e))?;
                eprintln!("Saved to {}", path.display());

                if codec.is_lossless() {
                    return Err(String::from("Decoded image doesn't match"))
                }
            }

            let mut lock_guard = writer_state.write().unwrap();
            let lock_guard = lock_guard.deref_mut();
            let wrote_header: &mut bool = &mut lock_guard.0;
            let csv: &mut _ = &mut lock_guard.1;

            if !*wrote_header {
                csv.write_record(&["name", "compressed_size", "compression_ratio", "error"])
                    .map_err(|e| format!("{:?}", e))?;
                *wrote_header = true;
            }

            let file_name = p.to_str().unwrap_or("???");
            csv.serialize((file_name, compressed_size, compression_ratio * 100.0, error))
                .map_err(|e| format!("{:?}", e))?;
            Ok(())
        })
        .for_each(|r| print_err(r));

    let (_wrote_header, mut csv) = writer_state.into_inner().unwrap();
    csv.flush()?;
    Ok(())
}

fn csv_writer<T: Codec>(codec: &T) -> Result<csv::Writer<fs::File>, csv::Error> {
    let mut csv_filename = String::from("output/");
    csv_filename.push_str(&codec.name());
    csv_filename.push_str(".csv");

    csv::Writer::from_path(&csv_filename)
}

// Compute MSE
// See https://stackoverflow.com/questions/20271479/what-does-it-mean-to-get-the-mse-mean-error-squared-for-2-images
fn compute_error(x: &Img, y: &Img) -> f64 {
    let tot_error: f64 = x.pixels()
                            .map(|tup| tup.2.to_rgb())
                            .zip(y.pixels()
                                    .map(|tup| tup.2.to_rgb()))
                            .map(|(px, py)| px.dist(&py).powi(2))
                            .sum();
    let (w, h) = x.dimensions();
    return tot_error / ((w * h) as f64);
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
