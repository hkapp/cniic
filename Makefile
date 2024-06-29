
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
CARGO_RUN = time cargo run --release --
DATASET = data/DIV2K_valid_HR/*

LOSSLESS_CODECS = $(HUFMAN) $(ZIP_DICT) $(ZIP_BACK_CP)
HUFMAN = output/Hufman.csv
ZIP_DICT = output/zip-dict.csv
# Slow codec: use '.cp' file (see rules below)
ZIP_BACK_ROOT = output/zip-back.csv
ZIP_BACK_CP = $(ZIP_BACK_ROOT).cp

LOSSY_CODECS = $(CLUSTER_COLORS) $(VORONOI)
# Slow codecs: use '.cp' files (see rules below)
CLUSTER_COLORS = output/cluster-colors_16.csv.cp output/cluster-colors_32.csv.cp output/cluster-colors_64.csv.cp \
	output/cluster-colors_128.csv.cp output/cluster-colors_256.csv.cp
VORONOI = output/voronoi_64.csv.cp	output/voronoi_128.csv.cp output/voronoi_256.csv.cp output/voronoi_512.csv.cp \
	output/voronoi_1024.csv.cp output/voronoi_2048.csv.cp

diagrams: $(TRACKED_DIAGRAMS)

lossless_status.png: $(LOSSLESS_DIAGRAM)
	cp $(LOSSLESS_DIAGRAM) lossless_status.png

lossy_status.png: $(LOSSY_DIAGRAM)
	cp $(LOSSY_DIAGRAM) lossy_status.png

PYTHON3 = PYTHONPATH="$(PYTHONPATH):scripts/" python3
SCRIPTS_LOC = scripts/diagrams

$(LOSSLESS_DIAGRAM): $(LOSSLESS_CODECS) $(SCRIPTS_LOC)/boxplot.py
	$(PYTHON3) $(SCRIPTS_LOC)/boxplot.py

$(LOSSY_DIAGRAM): $(LOSSLESS_CODECS) $(LOSSY_CODECS) $(SCRIPTS_LOC)/error_vs_compression_plot.py
	$(PYTHON3) $(SCRIPTS_LOC)/error_vs_compression_plot.py

$(HUFMAN):
	$(CARGO_RUN) --codec=hufman $(DATASET)

# For codecs that are too slow to compute, we simply keep a bak file locally
# and copy it if the original gets overwritten (e.g. via prepare_challenge.sh)
# The codecs that use this must provide a '.bak' rule WITHOUT dependencies on the base file
output/%.cp: output/%.bak
	cp output/$*.bak output/$*
	touch output/$*.cp

output/cluster-colors_%.csv.bak:
	$(CARGO_RUN) --codec="cluster-colors($*)" $(DATASET)
	cp output/cluster-colors_$*.csv output/cluster-colors_$*.csv.bak

output/voronoi_%.csv.bak:
	$(CARGO_RUN) --codec="voronoi($*)" $(DATASET)
	cp output/voronoi_$*.csv output/cluster-colors_$*.csv.bak

$(ZIP_DICT):
	$(CARGO_RUN) --codec="zip(dict)" $(DATASET)

$(ZIP_BACK_ROOT).bak:
	$(CARGO_RUN) --codec="zip(back)" $(DATASET)
	cp $(ZIP_BACK_ROOT) $(ZIP_BACK_ROOT).bak
