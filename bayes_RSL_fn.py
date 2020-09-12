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
import matlab_for_python

plt.style.use('seaborn-dark-palette')

class BayesRSL(BZ.bzBase):
    def __init__(self):
        super().__init__()
        self.log.setLevel('DEBUG')
        np.random.seed(10)
        self.path_mdat = op.join(self.path_Brsl, 'shared_data')
        self.tg_data   = loadmat(op.join(self.path_mdat, 'DATA.mat'))['DATA']
        self.N, self.K = self.tg_data.shape
        self.M         = np.sum(~np.isnan(self.tg_data).all(1)) # # of tgs with >0 datums
        self.D         = LoadTGs()(False) # just distance matrix for now

    def __call__(self, NN_burn=1e5, NN_post = 1e5, thin_period=1e2):
        """ Perform the actual simulations """
        #####################################################
        # Loop through the Gibbs sampler with Metropolis step
        #####################################################
        NN_burn_thin = NN_burn / thin_period
        NN_post_thin = NN_post / thin_period
        NN_thin      = int(NN_burn_thin + NN_post_thin)

        ## setup the hyper params, initial values, and selection matrices
        self.set_identity()
        HP        = self.set_hyperparms()
        init_vals = self.set_initial_vals(HP)
        y, y0, mu, nu, pi_2, delta_2, sigma_2, tau_2, phi, b, l, r = init_vals
        sel_mat_H, sel_mat_F, dct_z = self.set_selection()
        self.initialize_output(NN_thin)

        # time
        T     = np.arange(1.0, self.K+1)
        T    -= T.mean()
        T0    = T[0] - 1

        st, t_int = time.time(), time.time() # for logging
        NN    = NN_burn + NN_post
        for nn in np.arange(NN):
            if nn == 2:
                import ipdb
                from numpy import mean
                print (f'Step:',nn, 'y:', y.mean())
                print (f'Step:',nn, 'y0:', y0.mean())
                print (f'Step:',nn, 'b:', b.mean())
                print (f'Step:',nn, 'mu:', mu)
                print (f'Step:',nn, 'pi2:', pi_2)
                print (f'Step:',nn, 'delta2', delta_2)
                print (f'Step:',nn, 'r:', r)
                print (f'Step:',nn, 'sigma2:', sigma_2)
                print (f'Step:',nn, 'phi:', y.mean())
                print (f'Step:',nn, 'l:', l.mean())
                print (f'Step:',nn, 'nu:', nu)
                print (f'Step:',nn, 'tau2:', tau_2)
                ipdb.set_trace()

            if nn % 100 == 0 and not nn == 0:
                print (nn)
            #     elap  = time.time() - t_int
            #     self.log.info('%s of %s iterations in %.2f seconds', nn, NN, elap)
            #     t_int = time.time()

            nn_thin = int(np.floor(nn/thin_period))
            ####################################################################
            # Define matrices to save time
            ####################################################################
            # pi_2    = 1.528579081496371e-07 #mlab, step 20
            # sigma_2 = 5.906009002191849e-04 #mlab, step 20
            # phi     = 0.003041211145683

            BMat    = pi_2*np.eye(*self.D.shape)
            invBmat = np.linalg.inv(BMat)
            # if nn == 15:
                # print (phi, sigma_2, delta_2)
                # phi     = 0.0032
                # sigma_2 = 6.2300e-04
                # delta_2 = 3.0899e-05
                # return
            Sig     = sigma_2*np.exp(-phi*self.D)
            invSig  = np.linalg.inv(Sig)

            ####################################################################
            # Sample from p(y_K|.)
            ####################################################################
            # delta_2 = 2.770692212259742e-05
            # r       = 0.988291166084029
            # b  = loadmat(op.join(self.path_mdat, 'bl2'))['b'].squeeze() # mlab
            # l  = loadmat(op.join(self.path_mdat, 'bl2'))['l'].squeeze() # mlab

            K       = self.K-1
            V_Y_K   = (1/delta_2) * (sel_mat_H[K].T @ (dct_z[K] - sel_mat_F[K] @ l)) + \
                                  invSig@(r * y[:, K-1] + (T[K] - r * T[K-1]) * b)

            PSI_Y_K = (1/delta_2) * sel_mat_H[K].T @ sel_mat_H[K] + invSig
            # print (f'a {np.min(PSI_Y_K)}, {np.max(PSI_Y_K):.3f}')
            PSI_Y_K = np.linalg.matrix_power(PSI_Y_K, -1)
            # print ('b', np.min(PSI_Y_K), np.max(PSI_Y_K))
            # print (PSI_Y_K.T[:3,:3])

            y[:, K] = np.random.multivariate_normal(PSI_Y_K@V_Y_K, PSI_Y_K).T

            ## use the values from matlabs random generator; can skip and get them enxt
            # y[:, K] = matlab_for_python.y_K
            # y  = loadmat(op.join(self.path_mdat, 'y'))['y'] # mlab
            ####################################################################
            # Sample from p(y_k|.)
            ####################################################################
            # import ipdb
            # b = matlab_for_python.b
            # l = matlab_for_python.l
            # y0 = matlab_for_python.y0
            for kk in range(K-1, -1, -1):
                if kk == 0:
                    V_Y_k = 1/delta_2 * (sel_mat_H[kk].T @ (dct_z[kk] - sel_mat_F[kk]@l)) + \
                            invSig@(r*(y0 + y[:, kk+1]) + \
                            (1+r**2)*T[kk] * b - r*(T0 + T[kk+1])*b)
                else:
                    V_Y_k = 1/delta_2 * (sel_mat_H[kk].T @ \
                            (dct_z[kk] - sel_mat_F[kk]@l)) + \
                            invSig@(r*(y[:, kk-1] + y[:, kk+1]) + \
                            (1+r**2)*T[kk] * b - r*(T[kk-1] + T[kk+1])*b)

                PSI_Y_k = np.linalg.inv(1/delta_2*sel_mat_H[kk].T @ \
                                    sel_mat_H[kk]+(1+r**2)*invSig)

                # min_eig = np.min(np.real(np.linalg.eigvals(PSI_Y_k)))

                y[:, kk] = np.random.multivariate_normal(PSI_Y_k@V_Y_k, PSI_Y_k).T

            # y  = loadmat(op.join(self.path_mdat, 'y1'))['y'] # mlab

            ####################################################################
            # Sample from p(y0|.)
            ####################################################################
            V_Y_0   = (HP['eta_tilde_y0']/HP['delta_tilde_y0_2']) * self.ONE_N \
                                           + invSig@(r*(y[:,0])-r*(T[0]-r*T0)*b)
            PSI_Y_0 = np.linalg.inv(1/HP['delta_tilde_y0_2'] * self.I_N + r**2 * invSig)

            y0      = np.random.multivariate_normal(PSI_Y_0@V_Y_0, PSI_Y_0).T
            # y0 = loadmat(op.join(self.path_mdat, 'y_0'))['y_0'].squeeze() # mlab

            ####################################################################
            # Sample from p(b|.)
            ####################################################################
            SUM_K = self.ZERO_N
            # mu = 0.006894440462913 #mlab
            for kk in range(self.K):
                if kk == 0:
                    SUM_K += (T[0] - r * T0) * (y[:,kk] - r * y0)
                else:
                    SUM_K += (T[kk] - r*T[kk-1]) * (y[:, kk] - r * y[:, kk-1])
            V_B   = mu * invBmat @ self.ONE_N + invSig @ SUM_K
            tt    = np.array([T0, *T[0:self.K-1]])
            PSI_B = np.linalg.inv(invBmat + invSig * np.sum((T - r*tt)**2))
            b     = np.random.multivariate_normal(PSI_B@V_B, PSI_B).T
            # b     = b[..., np.newaxis] # force for linalg later; NOT NEEDED
            # b     = loadmat(op.join(self.path_mdat, 'b'))['b'].squeeze() # mlab

            ####################################################################
            # Sample from p(mu|.)
            ####################################################################
            V_MU   = HP['eta_tilde_mu']/HP['delta_tilde_mu_2'] + (self.ONE_N.T @ invBmat @ b)
            PSI_MU = 1/(1/HP['delta_tilde_mu_2'] + self.ONE_N.T @ invBmat @ self.ONE_N)
            mu     = np.random.normal(PSI_MU*V_MU, np.sqrt(PSI_MU))
            # mu = 0.006443022461383 # mlab

            ####################################################################
            # Sample from p(pi_2|.)
            ####################################################################
            inside1 = 1/2 * self.N
            soln    = np.linalg.lstsq(np.eye(*self.D.shape).T,
                                    b - mu * self.ONE_N, rcond=-1)[0]
            inside2 = 1/2 * soln @  (b - mu * self.ONE_N)
            pi_2    = 1/np.random.gamma(HP['lambda_tilde_pi_2']+inside1, 1/(HP['nu_tilde_pi_2']+inside2))

            ####################################################################
            # Sample from p(delta_2|.)
            ####################################################################
            SUM_K = 0
            for kk in range(self.K):
                xxx    = dct_z[kk] - sel_mat_H[kk] @ y[:, kk] - sel_mat_F[kk] @ l
                SUM_K += (xxx.T @ xxx)
            delta_2 = 1/np.random.gamma(HP['lambda_tilde_delta_2'] + 0.5*self.M_k.sum(), 1/(HP['nu_tilde_delta_2']+0.5*SUM_K))
            # delta_2 = 2.719951966190159e-04 # mlab

            ####################################################################
            # Sample from p(r|.)
            ####################################################################
            V_R, PSI_R = 0, 0
            for kk in range(self.K):
                if kk == 0:
                    V_R   += ((y0 - b * T0).T) @ invSig @ (y[:, kk] - b * T[kk])
                    PSI_R += ((y0 - b * T0).T) @ invSig @ (y0 - b * T0)
                else:
                    V_R   += ((y[:, kk-1] - b * T[kk-1]).T) @ invSig @ (y[:, kk] - b * T[kk])
                    PSI_R += ((y[:, kk-1] - b * T[kk-1]).T) @ invSig @ (y[:, kk-1] - b * T[kk-1])
            PSI_R = PSI_R**(-1)
            dummy = 1
            while dummy:
                sample = np.random.normal(PSI_R * V_R, np.sqrt(PSI_R))
                if HP['u_tilde_r'] < sample < HP['v_tilde_r']:
                    r     = sample
                    # r     = 0.957674212154219 # mlab
                    dummy = 0

            ####################################################################
            # Sample from p(sigma_2|.)
            ####################################################################
            # phi = 3.4538e-04
            RE, SUM_K = np.exp(-phi * self.D), 0
            invRE     = np.linalg.lstsq(RE, self.I_N, rcond=-1)[0]

            for kk in range(self.K):
                if kk == 0:
                    DYKK = y[:, kk] - r * y0 - (T[kk] - r * T0) * b
                else:
                    DYKK   = y[:, kk] - r * y[:, kk-1] - (T[kk] - r * T[kk-1]) * b

                SUM_K += DYKK.T @ invRE @ DYKK
            sigma_2 = 1/np.random.gamma((HP['lambda_tilde_sigma_2']+(self.N*self.K/2)), 1/(HP['nu_tilde_sigma_2']+(0.5*SUM_K)))#.squeeze()
            # sigma_2 = 0.002204008033078 # mlab


            ####################################################################
            # Sample from p(phi|.)
            ####################################################################
            Phi_now  = np.log(phi)
            Phi_std  = 0.05
            Phi_prp  = np.random.normal(Phi_now, Phi_std)
            # Phi_prp  = -5.589928909836823 # mlab
            R_now    = np.exp(-np.exp(Phi_now)*self.D)
            R_prp    = np.exp(-np.exp(Phi_prp)*self.D)
            invR_now = np.linalg.inv(R_now)
            invR_prp = np.linalg.inv(R_prp)

            sumk_now = 0
            sumk_prp = 0
            for kk in range(self.K):
                if kk == 0:
                    DYYK = y[:, kk] - r * y0 - (T[kk] - r * T0) * b
                else:
                    DYYK = y[:, kk] - r * y[:, kk-1] - (T[kk] - r * T[kk-1]) * b
                sumk_now += DYYK.T @ invR_now @ DYYK
                sumk_prp += DYYK.T @ invR_prp @ DYYK
            ins_now = -1 / (2*HP['delta_tilde_phi_2']) * (Phi_now - HP['eta_tilde_phi'])**2 - 1/(2*sigma_2) * sumk_now
            ins_prp = -1 / (2*HP['delta_tilde_phi_2']) * (Phi_prp - HP['eta_tilde_phi'])**2 - 1/(2*sigma_2) * sumk_prp
            MetFrac = np.linalg.det(R_prp@invR_now)**(-self.K/2) * np.exp(ins_prp-ins_now)

            success_rate = np.min([1, MetFrac])
            if np.random.uniform(1) <= success_rate:
                Phi_now = Phi_prp
            phi = np.exp(Phi_now)

            ####################################################################
            # Sample from p(l|.)
            ####################################################################
            SUM_K1, SUM_K2 = self.ZERO_M.copy(), np.zeros([self.M, self.M])
            for kk in range(self.K):
                SUM_K1 += (sel_mat_F[kk].T @ (dct_z[kk]-sel_mat_H[kk] @ y[:,kk]))
                SUM_K2 += (sel_mat_F[kk].T @ sel_mat_F[kk])

            V_L   = nu / tau_2 * self.ONE_M + 1/delta_2 * SUM_K1
            PSI_L = np.linalg.inv(1 / tau_2 * self.I_M + 1/delta_2 * SUM_K2)

            l  = np.random.multivariate_normal(PSI_L@V_L, PSI_L).T
            # l  = loadmat(op.join(self.path_mdat, 'l'))['l'].squeeze() # mlab

            ####################################################################
            # Sample from p(nu|.)
            ####################################################################
            V_NU = HP['eta_tilde_nu'] / HP['delta_tilde_nu_2'] + (1/tau_2 * (self.ONE_M.T @ l))
            PSI_NU = (1/HP['delta_tilde_nu_2'] + self.M / tau_2) ** -1
            nu     = np.random.normal(PSI_NU * V_NU, np.sqrt(PSI_NU))
            # nu     = 6.449444987285013 # mlab

            ####################################################################
            # Sample from p(tau_2|.)
            ####################################################################
            tau_2 = 1/np.random.gamma(HP['lambda_tilde_tau_2'] + self.M / 2,
                    1/(HP['nu_tilde_tau_2'] + 0.5*(((l-nu*self.ONE_M).T) @ (l-nu*self.ONE_M))))

            ####################################################################
            # Now update arrays
            ####################################################################
            self.MU[nn_thin]      = mu
            self.NU[nn_thin]      = nu
            self.PI_2[nn_thin]    = pi_2
            self.DELTA_2[nn_thin] = delta_2
            self.SIGMA_2[nn_thin] = sigma_2
            self.TAU_2[nn_thin] = tau_2
            self.PHI[nn_thin]   = phi
            self.B[nn_thin,:]   = b
            self.L[nn_thin,:]   = l
            self.R[nn_thin]     = r
            self.Y0[nn_thin,:] = y0
            self.Y[nn_thin,:,:] = y.T

        ##################################
        # delete the burn-in period values
        ##################################
        self.delete_burn_in(NN_burn_thin)

        #############
        # save output
        #############
        arrs2save = [self.MU, self.NU, self.PI_2, self.DELTA_2, self.SIGMA_2, self.TAU_2,
                     self.PHI, self.B, self.L, self.R, self.Y0, self.Y, self.tg_data, self.N, self.K, self.D]
        self.save_data(arrs2save, HP, save_tag=0)
        return

    def set_hyperparms(self):
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
            y           = self.tg_data[n, :]
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
        HP['delta_tilde_nu_2'] = var_infl * np.nanvar(l, ddof=1)  # var of nu prior

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

        return HP

    def set_initial_vals(self, HP):
        import scipy.stats as ss
        # mean parameters
        mu=np.random.normal(HP['eta_tilde_mu'], np.sqrt(HP['delta_tilde_mu_2']))
        nu=np.random.normal(HP['eta_tilde_nu'], np.sqrt(HP['delta_tilde_nu_2']))

        # variance parameters # use min to prevent needlessly large values
        pi_2    = np.min([1, 1/np.random.gamma(HP['lambda_tilde_pi_2'], 1/HP['nu_tilde_pi_2'])])
        delta_2 = np.min([1, 1/np.random.gamma(HP['lambda_tilde_delta_2'], 1/HP['nu_tilde_delta_2'])])
        sigma_2 = np.min([1, 1/np.random.gamma(HP['lambda_tilde_sigma_2'], 1/HP['nu_tilde_sigma_2'])])
        tau_2   = np.min([1, 1/np.random.gamma(HP['lambda_tilde_tau_2'], 1/HP['nu_tilde_tau_2'])])

        # inverse length scale parameters
        phi = np.exp(np.random.normal(HP['eta_tilde_phi'],np.sqrt(HP['delta_tilde_phi_2'])))

        ## temporarily overwriting to match piecuch
        # self.log.debug('Using piecuch initial vals')
        # mu, nu, pi_2, delta_2 =  0.009920200483713,  6.441041342905075, 1.247199778877984e-6, 9.476780305002085e-5 # mlab
        # sigma_2, tau_2, phi   = 0.001611436140762, 0.001773953960155, 0.003665877783146 # mlab

        # spatial fields
        b  = np.random.multivariate_normal(mu*np.ones(self.N), pi_2*np.eye(self.N))
        l  = np.random.multivariate_normal(nu*np.ones(self.N), tau_2*np.eye(self.N))

        # AR[1] parameter; drawn from uniform to maintain stationarity of time series
        r  = HP['u_tilde_r']+(HP['v_tilde_r']-HP['u_tilde_r'])*np.random.rand()

        # process
        y0  = np.zeros(self.N)
        y   = np.zeros([self.N,self.K])

        ## overwriting again;
        # matdat = loadmat(op.join(self.path_mdat, 'BL'))
        # b, l = matdat['b'].squeeze(), matdat['l'].squeeze() # mlab
        # r    = 0.495048630882454 # mlab

        return y, y0, mu, nu, pi_2, delta_2, sigma_2, tau_2, phi, b, l, r

    def set_selection(self):
        """ Setup selection and data matrices/dicts """
        ## make a table, each row is a tide gauge; eventually use tg names
        df_data  = pd.DataFrame(self.tg_data)
        H_master = ~np.isnan(self.tg_data)
        M_k      = H_master.sum(axis=0)
        dct_sel  = {'H': [], 'F': []}
        dct_z    = {}

        ## go through each column (year) and get the time series with data in it
        for i, col in enumerate(df_data.columns):
            tgs_with_data = df_data[col].dropna()
            tmp_h = np.zeros([M_k[i], self.N])
            tmp_f = np.zeros([M_k[i], self.M])
            for m in range(M_k[i]):
                tmp_h[m, tgs_with_data.index[m]] = 1
                tmp_f[m, tgs_with_data.index[m]] = 1
            dct_sel['H'].append(tmp_h)
            dct_sel['F'].append(tmp_f)
            if not np.allclose(tmp_h, tmp_f):
                self.log.warning('Tmp_H and tmp_F are not the same?')
            ## add an array of good values for each gauge
            dct_z[i] = tgs_with_data.values
        return dct_sel['H'], dct_sel['F'], dct_z

    def set_identity(self):
        """ Setup identity matrics and vectors of zeros/ones """
        # next two are duplicates in set_selection_dct
        H_master    = ~np.isnan(self.tg_data)
        self.M_k    = H_master.sum(axis=0)
        self.I_N    = np.eye(self.N)
        self.I_M    = np.eye(self.M)
        self.ONE_N  = np.ones([self.N])
        self.ONE_M  = np.ones([self.M])
        self.ZERO_N = np.zeros([self.N])
        self.ZERO_M = np.zeros([self.M])
        self.I_MK, self.ONE_MK, self.ZERO_MK = {}, {}, {}
        for k in range(self.K):
            self.I_MK[k]    = np.eye(self.M_k[k])
            self.ONE_MK[k]  = np.ones([self.M_k[k],1])
            self.ZERO_MK[k] = np.zeros([self.M_k[k],1])
        return

    def initialize_output(self, NN_thin):
        ########################################################################
        # initialize_output
        ########################################################################
        self.B = np.full([NN_thin,self.N], np.nan)          # Spatial field of process linear trend
        self.L = np.full([NN_thin,self.N], np.nan)          # Spatial field of observational biases
        self.R = np.full([NN_thin, 1], np.nan)              # AR(1) coefficient of the process
        self.Y = np.full([NN_thin, self.K, self.N], np.nan) # Process values
        self.Y0  = np.full([NN_thin, self.N], np.nan)      # Process initial conditions
        self.MU   = np.full([NN_thin,1], np.nan)      # Mean value of process linear trend
        self.NU   = np.full([NN_thin,1], np.nan)      # Mean value of observational biases
        self.PHI  = np.full([NN_thin,1], np.nan)      # Inverse range of process innovations
        self.PI_2 = np.full([NN_thin,1], np.nan)      # Spatial variance of process linear trend
        self.SIGMA_2 = np.full([NN_thin,1], np.nan)   # Sill of the process innovations
        self.DELTA_2 = np.full([NN_thin,1], np.nan)   # Instrumental error variance
        self.TAU_2   = np.full([NN_thin,1], np.nan)   # Spatial variance in observational biases

    def delete_burn_in(self, NN_burn_thin):
        NN_burn_thin = int(NN_burn_thin)
        self.B = self.B[NN_burn_thin:, :]
        self.L = self.L[NN_burn_thin:,:]
        self.R = self.R[NN_burn_thin:,:]
        self.Y = self.Y[NN_burn_thin:, :, :]
        self.Y0 = self.Y0[NN_burn_thin:,:]
        self.MU  = self.MU[NN_burn_thin:,:]
        self.NU  = self.NU[NN_burn_thin:,:]
        self.PHI = self.PHI[NN_burn_thin:,:]
        self.PI_2    = self.PI_2[NN_burn_thin:,:]
        self.SIGMA_2 = self.SIGMA_2[NN_burn_thin:,:]
        self.DELTA_2 = self.DELTA_2[NN_burn_thin:,:]
        self.TAU_2   = self.TAU_2[NN_burn_thin:,:]

    def save_data(self, arrs2save, HP, save_tag=0):
        import h5py, pickle
        path_save = op.join(self.path_Brsl, 'bayes_model_solution')
        os.makedirs(path_save, exist_ok=True)
        dst_h5    = op.join(path_save, f'py_exp{save_tag}.h5')
        arrnames  = 'MU NU PI_2 DELTA_2 SIGMA_2 TAU_2 PHI B L R Y_O Y TGDATA N K D'.split()
        with h5py.File(dst_h5, 'w') as h5:
            for arr, name in zip(arrs2save, arrnames):
                h5.create_dataset(name, data=arr)

        dst_dct   = f'{op.splitext(dst_h5)[0]}.dct'
        with open(dst_dct, 'wb') as fh:
            pickle.dump(HP, fh)

        self.log.info('Wrote results to:%s', dst_h5)
        self.log.info('Wrote results to:%s', dst_dct)

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
