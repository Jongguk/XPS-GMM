import numpy as np

random_state = np.random
X = np.concatenate([random_state.normal(-0.8, 2, 550),
                    random_state.normal(0.3, 2, 300), 
                    random_state.normal(2, 1.5, 150)])

print(X)

Y = X.reshape(-1,1)

print(Y)