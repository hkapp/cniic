mod bench;
mod bit;
mod huf;
mod kmeans;
mod utils;
mod ser;
mod codec;

fn main() {
    // Bash performs '*' expansion
    let file_paths = std::env::args().skip(1);
    let codec = codec::RedColKM(4096);
    bench::measure_all(&codec, file_paths).unwrap();
}
