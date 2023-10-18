use std::fs::{self, File};
use std::io;
use std::fmt::Debug;
use std::path::{Path, PathBuf};

pub type Img = image::DynamicImage;

pub trait Bench {
    fn encode<W: io::Write>(img: &Img, writer: &mut W);
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
            T::encode(&img, &mut data);

            let compressed_size = data.len();
            println!("bench: Compressed size = {} bytes", compressed_size);

            let decoded = T::decode(&mut data.into_iter());
            match decoded {
                Some(test) => {
                    if test != img {
                        use image::GenericImageView;
                        let mut px_pairs = test.pixels().zip(img.pixels());
                        let first_difference = px_pairs.find(|(enc_px, exp_px)| enc_px != exp_px);
                        eprintln!("First difference found: {:?}", first_difference);
                        eprintln!("Follow {} differences", px_pairs.count());

                        let mut second_encoding = Vec::new();
                        T::encode(&img, &mut second_encoding);

                        return Err(String::from("Decoded image doesn't match"))
                    }
                }
                None => {
                    return Err(String::from("Could not decode the image"))
                }
            }

            if !wrote_header {
                csv.write_record(&["name", "compressed_size"])
                    .map_err(|e| format!("{:?}", e))?;
                wrote_header = true;
            }

            let file_name = p.as_ref().to_str().unwrap_or("???");
            csv.serialize((file_name, compressed_size))
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
