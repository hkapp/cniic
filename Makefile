
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

LOSSLESS_DIAGRAM = output/boxplot.png
LOSSY_DIAGRAM = output/error_vs_compression.png

TRACKED_DIAGRAMS = lossy_status.png lossless_status.png

CODEC_DEPS = src/bench.rs
CARGO_RUN = cargo run --release --
DATASET = data/DIV2K_valid_HR/*

LOSSLESS_CODECS = $(HUFMAN) $(ZIP)
HUFMAN = output/Hufman.csv
ZIP = output/zip-dict.csv

LOSSY_CODECS =
# the following codecs are too slow to recompute every time:
# $(CLUSTER_COLORS) $(VORONOI)
CLUSTER_COLORS = output/cluster-colors_16.csv output/cluster-colors_32.csv output/cluster-colors_64.csv \
	output/cluster-colors_128.csv output/cluster-colors_256.csv
VORONOI = output/voronoi_64.csv	output/voronoi_128.csv output/voronoi_256.csv output/voronoi_512.csv \
	output/voronoi_1024.csv output/voronoi_2048.csv

diagrams: $(TRACKED_DIAGRAMS)

lossless_status.png: $(LOSSLESS_DIAGRAM)
	cp $(LOSSLESS_DIAGRAM) lossless_status.png

lossy_status.png: $(LOSSY_DIAGRAM)
	cp $(LOSSY_DIAGRAM) lossy_status.png

$(LOSSLESS_DIAGRAM): $(LOSSLESS_CODECS) boxplot.py
	python3 boxplot.py

$(LOSSY_DIAGRAM): $(LOSSLESS_CODECS) $(LOSSY_CODECS) error_vs_compression_plot.py
	python3 error_vs_compression_plot.py

$(HUFMAN): $(CODEC_DEPS) src/huf.rs src/codec/hufc.rs
	$(CARGO_RUN) --codec=hufman $(DATASET)

output/cluster-colors_%.csv: $(CODEC_DEPS) src/kmeans.rs src/codec/clusterc.rs
	$(CARGO_RUN) --codec="cluster-colors($*)" $(DATASET)

output/voronoi_%.csv: $(CODEC_DEPS) src/kmeans.rs src/codec/clusterc.rs
	$(CARGO_RUN) --codec="voronoi($*)" $(DATASET)

$(ZIP): $(CODEC_DEPS) src/zip.rs src/codec/zipc.rs
	$(CARGO_RUN) --codec="zip(dict)" $(DATASET)
