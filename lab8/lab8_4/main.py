import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results.csv")
for alg in ["naive", "row", "col", "block", "block_unroll"]:
    sub = df[(df.algorithm == alg) & (df.N == 1024)]
    plt.plot(sub.S, sub.gflops, marker="o", label=alg)
plt.xlabel("S")
plt.ylabel("GFLOPS")
plt.legend()
plt.show()
