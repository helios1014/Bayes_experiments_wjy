# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 15:24:34 2023

@author: gargo
"""

import arviz as az
import numpy as np
import pymc as pm
import scipy as sp
import seaborn as sns
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from statsmodels import datasets
from aesara import shared
from aesara import tensor as at

az.style.use("arviz-darkgrid")

sns.set()
blue, green, red, purple, gold, teal = sns.color_palette(n_colors=6)

pct_formatter = StrMethodFormatter("{x:.1%}")

#data from https://r-data.pmagunia.com/dataset/r-dataset-package-hsaur-mastectomy

df = pd.read_csv(r'C:\Users\gargo\Downloads\dataset-91668.csv',
                 dtype={'time':int,'event':int}, converters={ 'metastized': lambda x: {'yes':1,'no':0}[x]})   


df.head()

S0 = sp.stats.expon.sf

fig, ax = plt.subplots(figsize=(8, 6))

t = np.linspace(0, 10, 100)

ax.plot(t, S0(5 * t), label=r"$\beta^{\top} \mathbf{x} = \log\ 5$")
ax.plot(t, S0(2 * t), label=r"$\beta^{\top} \mathbf{x} = \log\ 2$")
ax.plot(t, S0(t), label=r"$\beta^{\top} \mathbf{x} = 0$ ($S_0$)")
ax.plot(t, S0(0.5 * t), label=r"$\beta^{\top} \mathbf{x} = -\log\ 2$")
ax.plot(t, S0(0.2 * t), label=r"$\beta^{\top} \mathbf{x} = -\log\ 5$")

ax.set_xlim(0, 10)
ax.set_xlabel(r"$t$")

ax.yaxis.set_major_formatter(pct_formatter)
ax.set_ylim(-0.025, 1)
ax.set_ylabel(r"Survival probability, $S(t\ |\ \beta, \mathbf{x})$")

ax.legend(loc=1)
ax.set_title("Accelerated failure times");



n_patient, _ = df.shape

X = np.empty((n_patient, 2))
X[:, 0] = 1.0
X[:, 1] = df.metastized


import pytensor.tensor as tt

VAGUE_PRIOR_SD = 5.0

with pm.Model() as weibull_model:
    β = pm.Normal("β", 0.0, VAGUE_PRIOR_SD, shape=2)

    

    
X_ = shared(X.astype(int))

with weibull_model:
    η = β.dot(X_.get_value().T) 
    
with weibull_model:
    s = pm.HalfNormal("s", 5.0)    

y = np.log(df.time.values)
y_std = (y - y.mean()) / y.std()

cens = df.event.values == 0.0


cens_ = shared(cens)



with weibull_model:
    y_obs = pm.Gumbel("y_obs", η[~cens_.get_value()], s, observed=y_std[~cens])


def gumbel_sf(y, μ, σ):
    return 1.0 - tt.exp(-tt.exp(-(y - μ) / σ)) 

with weibull_model:
    y_cens = pm.Potential("y_cens", gumbel_sf(y_std[cens], η[cens_.get_value()], s))

SEED = 845199  # from random.org, for reproducibility

SAMPLE_KWARGS = {"chains": 3, "tune": 1000, "random_seed": [SEED, SEED + 1, SEED + 2]}

with weibull_model:
    weibull_trace = pm.sample(**SAMPLE_KWARGS)    
    

az.plot_energy(weibull_trace);

x_test = shared(np.random.randn(3,4), shape=(3,4)) #see email
test2 = tt.as_tensor_variable(X.astype(int), dtype="int32")  #creates a tensor variabel but the transpose does not work?

https://www.pymc.io/projects/examples/en/latest/case_studies/reinforcement_learning.html

https://www.pymc.io/projects/examples/en/latest/howto/data_container.html

 Starting with PyMC 5.0, Aesara was replaced by PyTensor (see https://www.pymc.io/blog/pytensor_announcement.html).   
 
 Replace your import of aesara.tensor with pytensor.tensor.