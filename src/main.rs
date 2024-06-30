mod bench;
mod bit;
mod codec;
mod geom;
mod huf;
mod hilbert;
mod kmeans;
mod prs;
mod ser;
mod utils;
mod zip;

use codec::AnyCodec;
use image::{Pixel, Rgba};
use std::{path::{Path, PathBuf}, str::FromStr};

fn main() {
    try_special()
        .or_else(|_|try_bench())
        .unwrap()
}

fn try_special() -> Result<(), Err> {
    fn retrieve_command(argument: &str) -> Result<&str, Err> {
        argument.strip_prefix("--special=")
                .ok_or_else(usage)
    }

    let (command, file_paths) = separate_args()?;
    match retrieve_command(&command)? {
        "hilbert" => {
            file_paths
                .for_each(|img_path| {
                    fn write_to_csv<I: Iterator<Item = Rgba<u8>>>(iter: I, img_path: &str, meth_name: &str) {
                        let file_suffix = format!("{}.hilbert.csv", meth_name);
                        let output_path = under_output(&img_path, &file_suffix);
                        let mut csv = csv::Writer::from_path(&output_path).unwrap();
                        csv.write_record(&["red", "blue", "green"]).unwrap();

                        iter.for_each(|px| csv.serialize(px.to_rgb().0).unwrap());
                    }

                    let img = image::open(&img_path).unwrap();

                    write_to_csv(hilbert::linearize_rect(&img), &img_path, "rect");
                    write_to_csv(hilbert::linearize_small(&img), &img_path, "small");
                    write_to_csv(hilbert::linearize_large(&img), &img_path, "large");
                })
        }
        s@_ => {
            return Err(format!("Invalid special command: {:?}", s));
        }
    }
    Ok(())
}

fn try_bench() -> Result<(), Err> {
    fn parse_codec(input: &str) -> Result<AnyCodec, Err> {
        let codec_str = input.strip_prefix("--codec=")
                            .ok_or_else(usage)?;

        AnyCodec::from_str(codec_str)
            .map_err(|e| format!("Malformed codec argument\n{:?}", e))
    }

    let (command, file_paths) = separate_args()?;
    let codec = parse_codec(&command)?;
    bench::measure_all(&codec, file_paths).unwrap();
    Ok(())
}

type Err = String;

fn separate_args() -> Result<(String, impl Iterator<Item=String>), Err> {
    // Note: Bash performs '*' expansion, so we always get a list here
    let mut args = std::env::args().skip(1);

    let command = args.next()
                    .ok_or_else(usage)?;

    Ok((command, args))
}

fn usage() -> String {
    String::from("Usage: cniic --codec=<codec> [<img file>..]
Available codecs:
  hufman
  reduce-colors(<ncolors>)")
}

fn under_output<P: AsRef<Path>>(p: &P, new_extension: &str) -> PathBuf {
    let mut path = PathBuf::from("output");
    path.push(p.as_ref().file_name().unwrap());
    path.set_extension(new_extension);
    return path;
}
