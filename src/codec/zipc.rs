use std::str::FromStr;

use image::{GenericImageView, Pixel};

use crate::{prs, ser::{Deserialize, SerStream}, zip::{zip_dict_decode, zip_dict_encode, zip_back_decode, zip_back_encode}};

use super::Codec;

pub enum Zip {
    Dict,
    Back
}

impl Codec for Zip {
    fn encode<W: std::io::Write>(&self, img: &super::Img, writer: &mut W) -> std::io::Result<()> {
        let bytes =
            SerStream::from_value(img.dimensions())
                .chain(SerStream::from_iter(img.pixels()
                            .map(|(_, _, rgba)| rgba.to_rgb())));

        match self {
            Zip::Dict => zip_dict_encode(bytes, writer),
            Zip::Back => zip_back_encode(bytes, writer),
        }
    }

    fn decode<I: Iterator<Item = u8>>(&self, reader: &mut I) -> Option<super::Img> {
        fn rebuild_image<I: Iterator<Item = u8>>(mut decoded_bytes: I) -> Option<super::Img> {
            let dims: (u32, u32) = Deserialize::deserialize(&mut decoded_bytes)?;
            let mut img = image::RgbImage::new(dims.0, dims.1);

            for px in img.pixels_mut() {
                *px = Deserialize::deserialize(&mut decoded_bytes)?;
            }
            Some(img.into())
        }

        match self {
            Zip::Dict => {
                let zip_decoded_bytes = zip_dict_decode(reader);
                rebuild_image(zip_decoded_bytes)
            }
            Zip::Back => {
                let zip_decoded_bytes = zip_back_decode(reader);
                rebuild_image(zip_decoded_bytes)
            }
        }
    }

    fn name(&self) -> String {
        match self {
            Zip::Dict => String::from("zip-dict"),
            Zip::Back => String::from("zip-back"),
        }
    }

    fn is_lossless(&self) -> bool {
        true
    }
}

impl FromStr for Zip {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (zip_name, zip_args) = prs::fun_call(s)
                                        .ok_or_else(|| String::from("Incorrect syntax"))?;
        let _ =  prs::matches_fully(zip_name, "zip")
                    .ok_or_else(|| format!("Incorrect name: {}", zip_name))?;

        if zip_args.len() != 1 {
            return Err(format!("Wrong number of arguments: expected 1, found {}", zip_args.len()));
        }

        match zip_args[0] {
            "dict" => Ok(Zip::Dict),
            "back" => Ok(Zip::Back),
            a@_ => Err(format!("Unrecognized argument: {}", a)),
        }
    }
}
