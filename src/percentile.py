import pandas as pd
import numpy as np
import input_parse as ip
import matplotlib.pyplot as plt
from columns import get_columns


def get_percentile_density() -> list[float]:
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def plot_percentiles(df: pd.DataFrame) -> None:
    df.quantile(get_percentile_density()).plot(
        subplots=True, layout=(5, 3), figsize=(20, 20), grid=True, legend=True
    )
    plt.grid()
    plt.savefig("./metadata/whole_dataset/percentiles.png")


if __name__ == "__main__":
    df = pd.read_excel("./metadata/clustered.xlsx")

    print(df)
    print(df.quantile(get_percentile_density()).unstack())
    df = df.loc[:, get_columns()]
    plot_percentiles(df)
