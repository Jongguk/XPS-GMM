# Author: Jong-Guk Ahn
from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

X = np.loadtxt('data/converted_raw.txt').reshape(-1,1) # 1차원 배열인 Load 데이터를 불러옴
# print(X)

XN = np.arange(2, 3)
X_models = [None for i in range(len(XN))]

for i in range(len(XN)):
    X_models[i] = GaussianMixture(XN[i]).fit(X)

x_AIC = [m.aic(X) for m in X_models]
print(x_AIC)

fig = plt.figure(figsize=(6, 4))
fig.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.93, wspace=0.5)
ax = fig.add_subplot(111) # (1x3) 짜리 그래프에 1번째 자리에 plot

X_M_best = X_models[np.argmin(x_AIC)] # argmin(AIC)

x = np.linspace(65, 75, 1000)
x_logprob = X_M_best.score_samples(x.reshape(-1,1))
x_responsibilities = X_M_best.predict_proba(x.reshape(-1, 1))
x_pdf = np.exp(x_logprob)
x_pdf_individual = x_responsibilities * x_pdf[:, np.newaxis]

ax.hist(X, 41, density=True, histtype='stepfilled', alpha=0.4)
ax.plot(x, x_pdf, '-k')
ax.plot(x, x_pdf_individual, '--k')
ax.set_xlabel('$x$')
ax.set_ylabel('$p(x)$')

x_p = x_responsibilities
x_p = x_p.cumsum(1).T

plt.show()