import pandas as pd
from PIL import Image
from .heatmap import Heatmapper
import numpy as np


def create_heatmap(df):
    img = Image.new("RGB", (1920, 1080))
    df = pd.DataFrame(df.values.repeat(df["duration"], axis=0), columns=df.columns)
    gaze_data = df.loc[:, ["x", "y"]].to_numpy()
    heatmapper = Heatmapper()
    heatmap = heatmapper.heatmap_on_img(gaze_data, img)
    heatmap = heatmap.resize((175, 90))
    heatmap = remove_alpha_channel(heatmap)
    return np.asarray(heatmap)


def create_heatmaps(df_list):
    heatmaps = np.array([create_heatmaps_frames(df) for df in df_list])
    print(heatmaps.shape)
    return heatmaps


def create_heatmaps_frames(df):
    number_of_frames = 30
    frames = np.array_split(df, number_of_frames)
    return [create_heatmap(frame) for frame in frames]


def remove_alpha_channel(image):
    background = Image.new("RGBA", image.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image)
    return alpha_composite.convert("RGB")
