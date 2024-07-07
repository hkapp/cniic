import os
import glob

def output_folder():
    # We expect that these scripts will always be called from
    # the root of the cniic project
    # So we use the cwd as start point for the path
    return os.path.join(os.getcwd(), "output")

def diagram_csvs():
    return [p for p in glob.glob(output_folder() + "/*.csv")
            if not p.endswith(".hilbert.csv")]
