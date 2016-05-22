import numpy as np
from statsmodels import distributions
from scipy.optimize import fsolve
from scipy import stats
from operator import sub
import math

__author__ = 'Denis Surzhko'

class LDPortfolio:
    """
    Basic functionality for all LDP calibration facilities.
    :attribute self.ar: Estimated Accuracy ratio given portfolio distribution and PD values
    :attribute self.ct: Central Tendency (mean PD) ratio given portfolio distribution and PD values
    """
    def __init__(self, portfolio, rating_type='RATING', pd_cnd=None):
        """
        :param portfolio: Unconditional portfolio distribution from the worst to the best credit quality;
        :param rating_type: In case RATING, each 'portfolio' item contains number of companies in a given rating class;
                            In case SCORE, each item in the 'portfolio' is an exact score;
        :param pd_cnd: Current conditional PD distribution from the worst to the best credit quality. Could be used
                            for current AR estimation.
        """
        self.portfolio = np.array(portfolio)
        self.pd_cnd = np.array(pd_cnd)
        self.rating_type = rating_type
        if rating_type == 'RATING':
            self.portfolio_size = self.portfolio.sum()
            self.portfolio_dist = self.portfolio.cumsum() / self.portfolio_size
            self.portfolio_dist = (np.hstack((0, self.portfolio_dist[:-1])) + self.portfolio_dist) / 2
            self.rating_prob = self.portfolio / self.portfolio_size
        else:
            self.portfolio_size = len(self.portfolio)
            portfolio_ecdf = distributions.ECDF(self.portfolio)
            self.portfolio_dist = np.minimum(portfolio_ecdf(self.portfolio), 1 - 1.0 / self.portfolio_size)
            self.rating_prob = np.repeat(1.0 / self.portfolio_size, self.portfolio_size)
        self.ct = None
        self.ar = None
        if not pd_cnd is None:
            self.ct, self.ar = self._ar_estimate(self.pd_cnd)

    def _ar_estimate(self, pd_cnd):
        ct = self.rating_prob.T.dot(pd_cnd)
        ar_1int_1 = self.rating_prob * pd_cnd
        ar_1int_1 = np.hstack((0, ar_1int_1[:-1]))
        ar_1int_1 =  (1 - pd_cnd) * self.rating_prob * ar_1int_1.cumsum()
        ar_1 = 2 * ar_1int_1.sum()
        ar_2 =  (1 - pd_cnd) * pd_cnd * self.rating_prob * self.rating_prob
        ar = (ar_1 + ar_2.sum()) * (1.0 / (ct * (1 - ct))) - 1
        return ct, ar.sum()

class QMM(LDPortfolio):
    """
    Calibrates conditional probabilities of default according to Quasi Moment Matching algorithm
    :attribute self.pd_cnd: calibrated conditional PD (on score/rating)
    :attribute self.alpha: intercept calibration parameter
    :attribute self.beta: slope calibration parameter
    """
    def __init__(self, portfolio, rating_type = 'RATING', clb_curve = 'robust.logit', portfolio_cnd_no_dft = None):
        """
        :param portfolio: Unconditional portfolio distribution from the worst to the best credit quality;
        :param rating_type: In case RATING, each 'portfolio' item contains number of companies in a given rating class;
                            In case SCORE, each item in the 'portfolio' is an exact score;
        :param clb_curve: In case ’logit’, simple logit calibration curve is used (is applicable only for
                        rating_type = ’SCORE’). In case ’robust.logit’, robust logit function is used
                        (see Tasche D.(2013) for details).
        :param portfolio_cnd_no_dft: conditional on no default portfolio distribution (in case None, unconditional
                                    portfolio distribution is used as a proxy)
        :return: initialized QMM class object
        """
        if rating_type == 'RATING' and clb_curve != 'robust.logit':
            raise ValueError('Simple logit calibration curve is applicable only for rating.type = \'SCORE\'')
        super().__init__(portfolio, rating_type, pd_cnd=None)
        if portfolio_cnd_no_dft is None:
            self.portfolio_cnd_no_dft = self.portfolio
        else:
            self.portfolio_cnd_no_dft = self.portfolio_cnd_no_dft
        self.clb_curve = clb_curve
        self.alpha = None
        self.beta = None

    def fit(self, ct_target, ar_target):
        """
        :param ct_target: target Central Tendency
        :param ar_target: target Accuracy Ratio
        :return: calibrated QMM class
        """
        a = self.__get_pd((0, 0))
        tf = lambda x: tuple(map(sub, self._ar_estimate(self.__get_pd(x)), (ct_target, ar_target)))
        params = fsolve(tf, (0, 0))
        self.alpha, self.beta = params
        self.pd_cnd = self.__get_pd(params)
        self.ct, self.ar = self._ar_estimate(self.pd_cnd)
        return self

    def __get_pd(self, params):
        if self.clb_curve == 'logit':
            return self._logit(self.portfolio, params)
        else:
            return self._robust_logit(self.portfolio_dist, params)

    @staticmethod
    def _robust_logit(x, params):
        alpha, beta = params
        return 1 / (1 + np.exp(- alpha - beta * stats.norm.ppf(x)))

    @staticmethod
    def _logit(x, params):
        alpha, beta = params
        return 1 / (1 + np.exp(- alpha - beta * x))


class VDB(LDPortfolio):
    """
    Calibrates conditional probabilities of default according to Van der Burgt, M. model
    :attribute self.pd_cnd: calibrated conditional PD (on score/rating)
    :attribute self.k: intercept calibration parameter
    """
    def __init__(self, portfolio, rating_type = 'RATING', pd_on_ar_estimation_sample=0.0):
        """
        :param portfolio: Unconditional portfolio distribution from the worst to the best credit quality;
        :param rating_type: In case RATING, each 'portfolio' item contains number of companies in a given rating class;
                            In case SCORE, each item in the 'portfolio' is an exact score;
        :param pd_on_ar_estimation_sample: mean portfolio PD on AR estimation sample
        :return: initialized VDB class object
        """
        super().__init__(portfolio, rating_type, pd_cnd=None)
        self.pd_on_ar_estimation_sample = pd_on_ar_estimation_sample

    def fit(self, ct_target, ar_target):
        """
        :param ct_target: target Central Tendency
        :param ar_target: target Accuracy Ratio
        :return: calibrated VDB class
        """
        self._k = self._vdb_get_k(ar_target)[0]
        self.pd_cnd = ct_target * self._vdb_cap_der(self.portfolio_dist)
        self.ct, self.ar = self._ar_estimate(self.pd_cnd)
        return self

    def _vdb_cap_der(self, x):
        return self._k * np.exp(-self._k * x)/(1 - math.exp(-self._k))

    def _vdb_ar(self, guess_k):
        return 2.0 * (1.0 / (1.0 - math.exp(-guess_k)) - 1.0 / guess_k - 0.5) / (1.0 - self.pd_on_ar_estimation_sample)

    def _vdb_get_k(self, ar_target):
        return fsolve(lambda k: self._vdb_ar(k) - ar_target, 1)



# Examples
if __name__ == "__main__":
    portf_scores = tuple(range(100))
    pd = [x / 1000 for x in range(100)]
    pd = [0.1 for x in range(100)]
    pd = pd[::-1]
    # Creating portfolio class in order to estimate AR and CT
    p = LDPortfolio(portf_scores, rating_type='SCORE', pd_cnd=pd)
    p1 = LDPortfolio([10,20,30,10], rating_type='RATING', pd_cnd=[0.4, 0.3, 0.2, 0.1])
    # Applying QMM model
    q = QMM(portf_scores, rating_type='SCORE')
    q.fit(ct_target=0.1, ar_target=0.3)
    print(q.alpha)
    print(q.beta)
    print(q.pd_cnd)
    q1 = QMM([10,20,30,10], rating_type='RATING')
    q1.fit(ct_target=0.1, ar_target=0.3)
    print(q1.alpha)
    print(q1.pd_cnd)
    print(q1.beta)
    # Applying VDB model
    v = VDB([50,20,100,10,10], rating_type='RATING', pd_on_ar_estimation_sample = 0.5)
    v.fit(ct_target=0.1, ar_target=0.5)
    print(v.pd_cnd)
    print(v.ct)
    print(v.ar)


