import os
import os.path as op
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import BZ
from BZ import bbTS, bbGIS
from scipy.io import loadmat
import spectrum
import time

plt.style.use('seaborn-dark-palette')

class BayesRSL(BZ.bzBase):
    def __init__(self):
        super().__init__()
        self.log.setLevel('DEBUG')
        np.random.seed(10)
        self.data      = loadmat(op.join(self.path_Brsl, 'DATA.mat'))['DATA']
        self.N, self.K = self.data.shape
        self.M         = np.sum(~np.isnan(self.data).all(1)) # # of tgs with >0 datums
        self.D         = LoadTGs()(False)

        self.set_hyperparms()
        self.set_initial_vals()
        self.set_selection_dct()
        self.set_identity()

    def __call__(self, NN_burn=1e5, NN_post = 1e5, thin_period=1e2):
        """ Perform the actual simulations """
        #####################################################
        # Loop through the Gibbs sampler with Metropolis step
        #####################################################
        NN_burn_thin = NN_burn / thin_period
        NN_post_thin = NN_post / thin_period
        NN_thin      = NN_burn_thin + NN_post_thin
        NN           = NN_burn + NN_post
        st           = time.time()
        t_int        = st
        K            = self.K - 1 # for correct indexing
        # time
        T            = np.arange(1.0, self.K+1)
        T           -= T.mean()

        for nn in np.arange(NN):
            if nn % 50 == 0 and not nn == 0:
                elap  = time.time() - t_int
                self.log.info('%s of %s iterations in %s seconds', nn, NN, t_int)
                t_int = time.time()

            nn_thin = np.ceil(nn/thin_period)

            ####################################################################
            # Define matrices to save time
            ####################################################################
            BMat    = self.pi_2*np.eye(self.D.size)
            invBmat = np.linalg.inv(BMat)
            Sig     = self.sigma_2*np.exp(-self.phi*self.D)
            invSig  = np.linalg.inv(Sig)


            ####################################################################
            # Sample from p(y_K|.)
            ####################################################################

            V_Y_K = (1/self.delta_2) * (self.dct_sel['H'][self.K-1].T @ \
                      (self.dct_z[K] - self.dct_sel['F'][K] @ self.l)) + \
                    invSig@(self.r * self.y[:, K-1] + (T[K] - self.r * T[K-1]) * self.b)

            print (V_Y_K.mean())
            return

            # invSig * (self.r * self.y[:, K-1] + ( self.T(K) - self.r*T(K-1)) * self.b)

            # *(Z(K).z-selection_matrix(K).F*(l)))+...
            # 	invSig*(r*y(:,K-1)+(T(K)-r*T(K-1))*b);
            # PSI_Y_K=(1/delta_2*selection_matrix(K).H'*selection_matrix(K).H+invSig)^(-1);
            # y(:,K)=mvnrnd(PSI_Y_K*V_Y_K,PSI_Y_K)';
            # clear V_Y_K PSI_Y_K
        return

    def set_hyperparms(self):
        # time = np.array(list(range(1, K+1)))
        time = np.array(list(range(1, self.K+1)))
        ## initialize arrays
        m   = np.full([self.N, 1], np.nan) # slope estimates
        s   = np.full([self.N, 1], np.nan) # paramater covariance estimates
        r   = np.full([self.N, 1], np.nan)
        e   = np.full([self.N, 1], np.nan)
        n   = np.full([self.N, 1], np.nan)
        l   = np.full([self.N, 1], np.nan) # mean fit estimate
        y0  = np.full([self.N, 1], np.nan)

        for n in range(self.N):
            ## calculate the process parameters by using trend diff
            y           = self.data[n, :]
            mask        = ~np.isnan(y)
            y, t        = y[mask], time[mask]
            coeffs, cov = np.polyfit(t, y, 1, cov=True)
            m[n]  = coeffs[0]
            s[n]  = cov[0,0]
            l[n]  = coeffs[0]*time.mean()+coeffs[1]
            y0[n] = coeffs[1]-l[n] # predicted mean - offset
            ## for residual between time series and fit, model as ar1 process ideally results in white noise
            a, b  = spectrum.aryule(y-coeffs[0] * t-coeffs[1], 1)[:2]
            r[n] = -a
            e[n] = np.sqrt(b)

        ## ddof needs to be 1 for unbiased
        ## variance inflation parameters (to expand priors)
        var_infl, var_infl2, var0_infl  = 5**2, 10**2, 1
        HP = {}
        # y0
        HP['eta_tilde_y0']     = np.nanmean(y0)                    # mean of y0 prior
        HP['delta_tilde_y0_2'] = var0_infl * np.nanvar(y0, ddof=1) # var of y0 prior
        # r
        HP['u_tilde_r']        = 0 # lower bound of r (uniform) prior
        HP['v_tilde_r']        = 1 # upper bound of r (uniform) prior

        # mu
        HP['eta_tilde_mu']     = np.nanmean(m)                    # mean of mu (rate) prior
        HP['delta_tilde_mu_2'] = var_infl2 * np.nanvar(m, ddof=1) # var of mu prior

        # nu
        HP['eta_tilde_nu']     = np.nanmean(l)                    # mean of nu (mean pred) prior
        HP['delta_tilde_nu_2'] = var_infl * np.nanvar(m, ddof=1)  # var of nu prior

        # pi_2
        HP['lambda_tilde_pi_2'] = 0.5                        # shape of pi_2 prior
        HP['nu_tilde_pi_2']     = 1/2 * np.nanvar(m, ddof=1) # inverse scale of pi_2 prior

        # delta_2
        HP['lambda_tilde_delta_2'] = 0.5        # Shape of delta_2 prior
        HP['nu_tilde_delta_2']     = 0.5 * 1e-4 # Guess (1 cm)^2 error variance

        # sigma_2
        HP['lambda_tilde_sigma_2'] = 0.5                    # Shape of sigma_2 prior
        HP['nu_tilde_sigma_2']     = 0.5 * np.nanmean(e**2) # Inverse scale of sigma_2 prior

        # tau_2
        HP['lambda_tilde_tau_2']   = 0.5                        # Shape of tau_2 prior
        HP['nu_tilde_tau_2']       = 0.5 * np.nanvar(l, ddof=1) # Inverse scale of tau_2 prior

        # phi
        HP['eta_tilde_phi']        = -7 # "Mean" of phi prior
        HP['delta_tilde_phi_2']    = 5  # "Variance" of phi prior

        self.HP = HP
        return

    def set_initial_vals(self):
        """ Put hyperpams in here? """
        import scipy.stats as ss
        # mean parameters
        self.mu=np.random.normal(self.HP['eta_tilde_mu'], np.sqrt(self.HP['delta_tilde_mu_2']))
        self.nu=np.random.normal(self.HP['eta_tilde_nu'], np.sqrt(self.HP['delta_tilde_nu_2']))

        # variance parameters # use min to prevent needlessly large values
        self.pi_2    = np.min([1, 1/np.random.gamma(self.HP['lambda_tilde_pi_2'], 1/self.HP['nu_tilde_pi_2'])])
        self.delta_2 = np.min([1, 1/np.random.gamma(self.HP['lambda_tilde_delta_2'], 1/self.HP['nu_tilde_delta_2'])])
        self.sigma_2 = np.min([1, 1/np.random.gamma(self.HP['lambda_tilde_sigma_2'], 1/self.HP['nu_tilde_sigma_2'])])
        self.tau_2   = np.min([1, 1/np.random.gamma(self.HP['lambda_tilde_tau_2'], 1/self.HP['nu_tilde_tau_2'])])

        # inverse length scale parameters
        self.phi = np.exp(np.random.normal(self.HP['eta_tilde_phi'],np.sqrt(self.HP['delta_tilde_phi_2'])))

        ## temporarily overwriting to match piecuch
        self.log.debug('Using piecuch initial vals')
        self.mu, self.nu, self.pi_2, self.delta_2 =  0.009920200483713,  6.441041342905075, 1.247199778877984e-6, 9.476780305002085e-5
        self.sigma_2, self.tau_2, self.phi        = 0.001611436140762, 0.001773953960155, 0.003665877783146

        # spatial fields
        self.b  = np.random.multivariate_normal(self.mu*np.ones(self.N), self.pi_2*np.eye(self.N))
        self.l  = np.random.multivariate_normal(self.nu*np.ones(self.N), self.tau_2*np.eye(self.N))

        # AR[1] parameter; drawn from uniform to maintain stationarity of time series
        self.r  = self.HP['u_tilde_r']+(self.HP['v_tilde_r']-self.HP['u_tilde_r'])*np.random.rand()

        # process
        self.y_0 = np.zeros(self.N)
        self.y   = np.zeros([self.N,self.K])

        ## overwriting again
        matdat         = loadmat(op.join(self.path_Brsl, 'BL'))
        self.b, self.l = matdat['b'].squeeze(), matdat['l'].squeeze()
        self.r         = 0.495048630882454

        return

    def set_selection_dct(self):
        """ Setup selection dictionary """
        ## make a table, each row is a tide gauge; eventually use tg names
        df_data  = pd.DataFrame(self.data)
        H_master = ~np.isnan(self.data)
        M_k      = H_master.sum(axis=0)
        self.dct_sel  = {'H': [], 'F': []}
        self.dct_z    = {}

        ## go through each column (year) and get the time series with data in it
        for i, col in enumerate(df_data.columns):
            tgs_with_data = df_data[col].dropna()
            tmp_h = np.zeros([M_k[i], self.N])
            tmp_f = np.zeros([M_k[i], self.M])
            for m in range(M_k[i]):
                tmp_h[m, tgs_with_data.index[m]] = 1
                tmp_f[m, tgs_with_data.index[m]] = 1
            self.dct_sel['H'].append(tmp_h)
            self.dct_sel['F'].append(tmp_f)
            if not np.allclose(tmp_h, tmp_f):
                print ('False')
            ## add an array of good values for each gauge
            self.dct_z[i] = tgs_with_data.values
        return

    def set_identity(self):
        """ Setup identity matrics and vectors of zeros/ones """
        # next two are duplicates in set_selection_dct
        H_master    = ~np.isnan(self.data)
        M_k         = H_master.sum(axis=0)
        self.I_N    = np.eye(self.N)
        self.I_M    = np.eye(self.M)
        self.ONE_N  = np.ones([self.N,1])
        self.ONE_M  = np.ones([self.M,1])
        self.ZERO_N = np.zeros([self.N,1])
        self.ZERO_M = np.zeros([self.M,1])
        self.I_MK, self.ONE_MK, self.ZERO_MK = {}, {}, {}
        for k in range(self.K):
            self.I_MK[k]    = np.eye(M_k[k])
            self.ONE_MK[k]  = np.ones([M_k[k],1])
            self.ZERO_MK[k] = np.zeros([M_k[k],1])
        return

class LoadTGs(BZ.bzBase):
    def __init__(self):
        super().__init__()
        self.path_root  = op.join(self.path_Brsl, 'rlr_annual')
        self.path_flist = op.join(self.path_root, 'filelist.txt')

    def __call__(self, overwrite=False):
        ## get station locations that are in bbox
        dst = op.join(self.path_root, 'tg_distances.npy')
        if op.exists(dst) and not overwrite:
            self.log.debug('Using existing tg distances')
            return np.load(dst)

        self.df_locs    = self.load_sta_ids()
        ## get station timeseries, keep only time series with at least n_data pts
        self.df_ts      = self.get_ts(min_data=25)
        ## now get only lat/lon for stations with enough data
        df_locs  = self.df_locs[self.df_locs.index.isin(self.df_ts.columns)]
        gdf_locs = bbGIS.df2gdf(df_locs)
        return self.calc_distances(gdf_locs, dst)

    def calc_distances(self, gdf_tgs, dst):
        """ Calculate distances between each TG, square matrix result """
        from geopy import distance
        distance.EARTH_RADIUS = 6378.137 # to match matlab
        self.log.info('Calculating distance between TGs...')

        arr_dists = np.zeros([len(gdf_tgs), len(gdf_tgs)])
        for i, pt in enumerate(gdf_tgs.geometry):
            if i % 10 == 0: self.log.info('Processing points around TG %s of %s', i, gdf_tgs.shape[0])
            for j, pt2 in enumerate(gdf_tgs.geometry):
                dist = distance.great_circle((pt.y, pt.x), (pt2.y, pt2.x)).km
                arr_dists[i, j] = dist

        np.save(dst, arr_dists)
        return arr_dists

    def get_ts(self, min_data=25):
        """ Too many years, but same number (also correct?) """
        from functools import reduce
        ## load the actual time series
        tg_ids   = self.df_locs.index.values

        path_rlr = op.join(self.path_root, 'data')
        lst_src  = [op.join(path_rlr, f'{i}.rlrdata') for i in tg_ids]
        lst_sers = []
        for path_tg in lst_src:
            df_tmp = self._load_TG(path_tg, min_data)
            if df_tmp.empty: continue
            lst_sers.append(df_tmp)

        ## merge series
        df_merged = reduce(lambda  left,right: pd.merge(left, right, left_index=True, right_index=True,
                                            how='outer'), lst_sers)

        return df_merged

    def load_sta_ids(self, SNWE=[35, 46.24, -80, -60]):
        """ So far only implenenting for distance calcs """
        cols = ['ID', 'lat', 'lon', 'sta', 'ccode', 'scode', 'quality']
        df   = pd.read_csv(self.path_flist, delimiter=';', names=cols, header=None).set_index('ID')
        ## get those on the east coast from PSMSL coastal code
        df   = df[df.ccode.isin(['960', '970'])].sort_values('lat')
        ## subset by bbox
        S, N, W, E = SNWE
        return df[((df.lat>=S) & (df.lat <=N) & (df.lon >=W) & (df.lon<=E))]

    def _load_TG(self, path, min_data=25):
        """ Load a single raw tide gauge from the rlrfiles and check for bad data """
        id     = op.splitext(op.basename(path))[0]

        cols   = ['time', id, 'n_missing', 'flag']
        df_tg  = pd.read_csv(path, delimiter=';', names=cols).set_index('time')

        ## some just randomly empty
        if df_tg.empty: return pd.DataFrame()
        ## convert to nan
        df_tg.where(df_tg[id] > -1e4, np.nan, inplace=True)
        return df_tg[id] if df_tg[id].dropna().shape[0] > min_data else pd.DataFrame()

if __name__ == '__main__':
    # LoadTGs()(True)
    BayesRSL()()
