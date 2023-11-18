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
    bench::measure_all::<_, _, codec::RedColKM>(file_paths).unwrap();
}
