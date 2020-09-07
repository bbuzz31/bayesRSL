import os
import os.path as op
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import BZ
import statsmodels.formula.api as smf
import scipy.stats as ss
plt.style.use('seaborn-dark-palette')

def load_data(overwrite=False):
    """ Load the data; dist: avg drive (yards), acc: % shots land on fairway """
    dst = op.join(op.expanduser('~'), 'Desktop', 'Coursera_golf.df')
    if op.exists(dst) and not overwrite:
        df = pd.read_pickle(dst)
    else:
        df = pd.read_csv('http://www.stat.ufl.edu/~winner/data/pgalpga2008.dat',
                       delim_whitespace=True, names=['dist', 'acc', 'sex'], header=None)
        df.to_pickle(dst)

    df_f = df[df['sex'] == 1]
    df_m = df[df['sex'] == 2]
    return df_f, df_m

def scat_plots():
    for df, sex in zip([df_f, df_m], ['Women', 'Men']):
        ax = df.plot.scatter(x='dist', y='acc')
        ax.set_title(sex)
        ax.set_xlabel('Drive Distance (yards)')
        ax.set_ylabel('% of drives landing on Fairway')

def main():
    # scat_plots()
    coeffs, cov = np.polyfit(df_f.dist, df_f.acc, 1, cov=True)
    sem         = np.sqrt(np.diag(cov))[0]
    predictee   = 260
    pred        = np.polyval(coeffs, predictee)
    model  = smf.ols('acc ~ dist' , data=df_f).fit()
    x      = df_f.dist


    # lci, uci = -0.344, -0.169 # marginal bounds copied from result.summary()
    for qt in [0.025, 0.975]:
        qtt = ss.t.ppf(qt, model.nobs - 2)
        print (pred-np.sqrt(model.scale) * qtt * np.sqrt(1+1/model.nobs+((predictee-x.mean())**2/(model.nobs-1)/np.var(x))))
    return

def main2():
    df = pd.read_csv('http://www.randomservices.org/random/data/Challenger2.txt', sep='\t')
    df.columns=['Temp', 'Dam']
    x  = df.Temp

    coeffs, residuals  = np.polyfit(df.Temp, df.Dam, 1, full=True)[:2]
    predictee   = 31

    # sem         = np.sqrt(np.diag(cov))[0]
    pred        = np.polyval(coeffs, predictee)
    model  = smf.ols('Dam ~ Temp' , data=df).fit()
    for qt in [0.025, 0.975]:
        qtt = ss.t.ppf(qt, model.nobs - 2)
        print (pred-np.sqrt(model.scale) * qtt * np.sqrt(1+1/model.nobs+((predictee-x.mean())**2/(model.nobs-1)/np.var(x))))

if __name__ == '__main__':
    df_f, df_m = load_data()
    main()
    plt.show()
