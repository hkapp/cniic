
all: data output data/DIV2K_valid_HR

clean:
	rm -r data output

data:
	mkdir data

output:
	mkdir output

# https://data.vision.ee.ethz.ch/cvl/DIV2K/
data/DIV2K_valid_HR.zip:
	wget -P data/ http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

data/DIV2K_valid_HR:
	unzip data/DIV2K_valid_HR.zip -d data/
