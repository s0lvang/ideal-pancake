import pandas as pd
import detectors, gazeplotter

df = pd.read_csv("80_rawdata.tsv", delimiter="\t", comment="#")
df = df[df["Type"] == "SMP"]

fixations = detectors.fixation_detection(df['L POR X [px]'].to_numpy(), df['L POR Y [px]'].to_numpy(), df['Time'].to_numpy(), maxdist=25, mindur=500)
print(fixations[1])
print(len(fixations[0]))
