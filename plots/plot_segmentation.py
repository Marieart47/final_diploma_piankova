import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("segmentation_results.csv")

for model in df.model.unique():
    plt.figure()
    for strat in df.strategy.unique():
        d = df[(df.model == model) & (df.strategy == strat)]
        plt.plot(d.epoch, d.miou, label=strat)
    plt.title(model)
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.legend()
    plt.savefig(f"{model}_miou.png")
