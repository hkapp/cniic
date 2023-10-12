use std::fs;
use std::io;
use std::fmt::Debug;
use std::path::PathBuf;

pub type Img = image::DynamicImage;

pub trait Bench {
    fn encode<W: io::Write>(img: &Img, writer: &mut W);
    fn decode<I: Iterator<Item = u8>>(reader: &mut I) -> Option<Img>;
    fn name() -> String;
}

pub fn measure_all<T: Bench>(input_data: &str) -> io::Result<()> {
    let mut wrote_header = false;
    let mut csv = csv_writer::<T>()?;

    std::fs::read_dir(input_data)?
        .map(|mbf| {
            let f = mbf.map_err(|e| format!("{:?}", e))?;
            let file_name = f.path();

            let img = load_image(&file_name)
                        .map_err(|e| format!("{:?}", e))?;

            let mut data = Vec::new();
            T::encode(&img, &mut data);

            let compressed_size = data.len();

            let decoded = T::decode(&mut data.into_iter());
            match decoded {
                Some(test) => {
                    if test != img {
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
            csv.serialize((file_name, compressed_size))
                .map_err(|e| format!("{:?}", e))?;
            Ok(())
        })
        .for_each(|r| print_err(r));

    csv.flush()?;
    Ok(())
}

fn csv_writer<T: Bench>() -> Result<csv::Writer<fs::File>, csv::Error> {
    let mut csv_filename = String::from("data/");
    csv_filename.push_str(&T::name());
    csv_filename.push_str(".csv");

    csv::Writer::from_path(&csv_filename)
}

fn load_image(path: &PathBuf) -> Result<Img, image::ImageError> {
    image::open(path)
}

fn print_err<T, E: Debug>(r: Result<T, E>) {
    match r {
        Err(e) => eprintln!("{:?}", e),
        _      => (),
    }
}
