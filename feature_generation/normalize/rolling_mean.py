def rolling_mean(data):
    return [take_rolling_mean(df) for df in data]


def take_rolling_mean(df):
    window = 100
    columns_to_take_rolling_mean = [
        "pupil_diameter",
        "saccade_duration",
        "duration",
        "saccade_length",
    ]
    for column in columns_to_take_rolling_mean:
        df[f"{column}_rolling"] = df[column].rolling(window).mean()
    # index < window is nan
    return df.iloc[window:]
