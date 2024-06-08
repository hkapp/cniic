mod bench;
mod bit;
mod codec;
mod geom;
mod huf;
mod kmeans;
mod utils;
mod ser;
mod zip;

use codec::AnyCodec;
use std::str::FromStr;

fn main() {
    let (codec, file_paths) = parse_args().unwrap();
    bench::measure_all(&codec, file_paths).unwrap();
}

type Err = String;

fn parse_args() -> Result<(AnyCodec, impl Iterator<Item=String>), Err> {
    // Note: Bash performs '*' expansion, so we always get a list here
    let mut args = std::env::args().skip(1);

    let codec = args.next()
                    .ok_or_else(usage)
                    .and_then(|s| parse_codec(&s))?;

    Ok((codec, args))
}

fn usage() -> String {
    String::from("Usage: cniic --codec=<codec> [<img file>..]
Available codecs:
  hufman
  reduce-colors(<ncolors>)")
}

fn parse_codec(input: &str) -> Result<AnyCodec, Err> {
    let codec_str = input.strip_prefix("--codec=")
                        .ok_or_else(usage)?;

    AnyCodec::from_str(codec_str)
        .map_err(|e| format!("Malformed codec argument\n{:?}", e))
}
