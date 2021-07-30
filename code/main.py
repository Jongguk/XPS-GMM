# Author: Jong-Guk Ahn
from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

random_state = np.random
X = np.concatenate([random_state.normal(-0.8, 2, 550),
                    random_state.normal(0.3, 2, 300), 
                    random_state.normal(2, 1.5, 150)]).reshape(-1, 1)
random_state = np.random
Y = np.concatenate([random_state.normal(-0.2, 1.2, 450),
                    random_state.normal(0, 1.4, 400), 
                    random_state.normal(2, 1, 350)]).reshape(-1, 1)

XN = np.arange(1, 11)
X_models = [None for i in range(len(XN))]

for i in range(len(XN)):
    X_models[i] = GaussianMixture(XN[i]).fit(X)

x_AIC = [m.aic(X) for m in X_models]
print(x_AIC)
# BIC = [m.bic(X) for m in models]

fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(left=0.12, right=0.97, bottom=0.21, top=0.9, wspace=0.5)
ax = fig.add_subplot(121) # (1x3) 짜리 그래프에 1번째 자리에 plot

X_M_best = X_models[np.argmin(x_AIC)] # argmin(AIC)

x =np.linspace(-10, 10, 1000)
x_logprob = X_M_best.score_samples(x.reshape(-1,1))
x_responsibilities = X_M_best.predict_proba(x.reshape(-1, 1))
x_pdf = np.exp(x_logprob)
x_pdf_individual = x_responsibilities * x_pdf[:, np.newaxis]

ax.hist(X, 30, density=True, histtype='stepfilled', alpha=0.4)
ax.plot(x, x_pdf, '-k')
ax.plot(x, x_pdf_individual, '--k')
ax.set_xlabel('$x$')
ax.set_ylabel('$p(x)$')

x_p = x_responsibilities
x_p = x_p.cumsum(1).T

YN = np.arange(1, 11)
Y_models = [None for i in range(len(YN))]

for i in range(len(YN)):
    Y_models[i] = GaussianMixture(YN[i]).fit(Y)

y_AIC = [m.aic(Y) for m in Y_models]

ay = fig.add_subplot(122) # (1x3) 짜리 그래프에 1번째 자리에 plot
Y_M_best = Y_models[np.argmin(y_AIC)]

y = np.linspace(-10, 10, 1000)
y_logprob = Y_M_best.score_samples(y.reshape(-1,1))
y_responsibilities = Y_M_best.predict_proba(y.reshape(-1, 1))
y_pdf = np.exp(y_logprob)
y_pdf_individual = y_responsibilities * y_pdf[:, np.newaxis]

ay.hist(Y, 30, density=True, histtype='stepfilled', alpha=0.4)
ay.plot(y, y_pdf, '-k')
ay.plot(y, y_pdf_individual, '--k')
ay.set_xlabel('$x$')
ay.set_ylabel('$p(x)$')

y_p = y_responsibilities
y_p = y_p.cumsum(1).T

plt.show()