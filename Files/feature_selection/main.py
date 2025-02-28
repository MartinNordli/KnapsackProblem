import numpy as np
import pandas as pd
import LinReg


if __name__ == "__main__":
    data = pd.read_csv("dataset.txt", header=None)
    # print(data)

    regressor = LinReg.LinReg()
    myRNG = np.random.default_rng()

    rand_ind = myRNG.integers(0, 1, size=data.shape[1], endpoint=True)
    X = regressor.get_columns(data.values, rand_ind)

    print(regressor.get_fitness(X[:,:-1], X[:,-1]))