import math
import collections

__author__ = 'Denis Surzhko'


class LDPortfolio:
    """
    Basic functionality for all LDP calibration facilities.
    """
    def __init__(self, portf_uncond, rating_type, pd_cond_old=None):
        """
        :param portf_uncond: Unconditional portfolio distribution from the worst to the best credit quality;
        :param rating_type: In case RATING, each portf_uncond item contains number of companies in a given rating class;
                            In case SCORE, each item in the portf_uncond is an exact score;
        :param pd_cond_old: Current conditional PD distribution from the worst to the best credit quality. Could be used
                            for current AR estimation.
        :return: Initialized LDPorfolio class.
        """
        self._portf_uncond = portf_uncond
        self._rating_type = rating_type
        self._pd_cond_old = pd_cond_old
        if pd_cond_old:
            self._ar_estimate()
        if rating_type == "RATING":
            portf_total = sum(portf_uncond)
            portf_cum = tuple(self._cumulative_sum(portf_uncond))
            portf_dist = tuple(map(sum, zip(portf_cum, (0,) + portf_cum[:len(portf_cum) - 1])))
            self._portf_dist = tuple((x / (2*portf_total)) for x in portf_dist)
        else:
            portf_total = len(portf_uncond)
            uniq_scores = collections.Counter(portf_uncond)
            portf_dist = tuple(uniq_scores[x] / portf_total for x in uniq_scores)
            self._portf_dist = tuple(self._cumulative_sum(portf_dist))

    def _ar_estimate(self):
        portf_total = len(self._portf_uncond)
        if self._rating_type == 'RATING':
            rating_prob = [x / portf_total for x in self._portf_uncond]
        else:
            rating_prob = [1 / portf_total] * portf_total
        self._current_ct = sum([a*b for a, b in zip(rating_prob, self._pd_cond_old)])
        ar_1int_1 = [a*b for a, b in zip(self._pd_cond_old, rating_prob)]
        ar_1int = tuple(self._cumulative_sum([0] + ar_1int_1[:-1]))
        ar_1 = 2 * sum([(1-a)*b*c for a, b, c in zip(self._pd_cond_old, rating_prob, ar_1int)])
        ar_2 = sum([(1-a)*a*b*b for a, b in zip(self._pd_cond_old, rating_prob)])
        self._implied_ar = (ar_1 + ar_2) * (1 / (self._current_ct * (1 - self._current_ct))) - 1

    @staticmethod
    def _cumulative_sum(values, start=0):
        for v in values:
            start += v
            yield start

    @property
    def portf_uncond(self):
        """
        :return: unconditional portfolio distribution
        """
        return self._portf_uncond

    @property
    def rating_type(self):
        """
        :return: RATING or SCORE type of the rating system
        """
        return self._rating_type

    @property
    def pd_cond_old(self):
        """
        :return: Current conditional PD distribution from the worst to the best credit quality.
        """
        return self._pd_cond_old

    @property
    def portf_dist(self):
        """
        :return: cumulative non-conditional portfolio distribution
        """
        return self._portf_dist

    @property
    def current_ct(self):
        """
        :return: current mean PD in the portfolio (aka central tendency)
        """
        return self._current_ct

    @property
    def implied_ar(self):
        """
        :return: current AR estimate based on provided conditional PD and portfolio distribution.
                 Estimation approach is based on:
                 Tasche, D. (2013) The art of probability-of-default curve calibration. Journal of Credit Risk, 9:63-103
        """
        return self._implied_ar


class VDBCalibration(LDPortfolio):
    """
    PD calibration according to Van der Burgt algorithm:
    Van der Burgt, M. (2008) Calibrating low-default portfolios, using the cumulative accuracy profile.
    Journal of Risk Model Validation, 1(4):17-33.
    """
    def __init__(self, portf_uncond, rating_type, target_ar, target_ct, pd_cond_old=None, pd_uncond_target_ar=0.0):
        """
        :param portf_uncond: Unconditional portfolio distribution from the worst to the best credit quality;
        :param rating_type: In case RATING, each portf_uncond item contains number of companies in a given rating class;
                            In case SCORE, each item in the portf_uncond is an exact score;
        :param target_ar: Target Accuracy Ratio for calibration purposes;
        :param target_ct: Target Central Tendency (mean PD in the portfolio) for calibration purposes;
        :param pd_cond_old: Current conditional PD distribution from the worst to the best credit quality. Could be used
                            for current AR estimation;
        :param pd_uncond_target_ar: Unconditional PD of the sample on which AR had been estimated
                                    (in case is zero, approximation AR = 2*AUC - 1 is used for parameters estimation).
        :return:
        """
        super().__init__(portf_uncond, rating_type, pd_cond_old)
        self._target_ar = target_ar
        self._target_ct = target_ct
        self._pd_uncond_target_ar = pd_uncond_target_ar
        self._k = self._vdb_get_k()
        self._pd_cond = tuple(target_ct * self._vdb_cap_der(x) for x in self.portf_dist)

    def _vdb_cap_der(self, x):
        return self._k * math.exp(-self._k * x)/(1 - math.exp(-self._k))

    def _vdb_ar(self, guess_k):
        return 2 * (1 / (1 - math.exp(-guess_k)) - 1 / guess_k - 0.5) / (1 - self.pd_uncond_target_ar)

    def _vdb_ar_der(self, guess_k):
        return 2 * (-math.pow(1 - math.exp(-guess_k), -2) *
                    math.exp(-guess_k) + math.pow(guess_k, -2)) / (1 - self.pd_uncond_target_ar)

    def _vdb_get_k(self):
        h = 0.00001
        last_x = h
        next_x = last_x + 10*h
        i = 0
        while abs(last_x - next_x) > h and i < 1000:
            last_x = next_x
            next_x = last_x - (self._vdb_ar(last_x) - self._target_ar) / self._vdb_ar_der(last_x)
            i += 1
        return last_x

    @property
    def target_ar(self):
        """
        :return: Target Accuracy Ratio that was used in calibration process
        """
        return self._target_ar

    @property
    def target_ct(self):
        """
        :return: Target Central Tendency (mean PD) that was used in calibration process
        """
        return self._target_ct

    @property
    def k(self):
        """
        :return: Smooth parameter for CAP  curve.
        """
        return self._k

    @property
    def pd_cond(self):
        """
        :return: Calibrated Probabilities of Default
        """
        return self._pd_cond

    @property
    def pd_uncond_target_ar(self):
        """
        :return: Unconditional PD of the sample on which AR had been estimated
                 (in case is zero, approximation AR = 2*AUC - 1 was used for parameters estimation).
        """
        return self._pd_uncond_target_ar

# Examples
# portf_scores = tuple(range(100))
# pd = [x / 1000 for x in range(100)]
# pd = pd[::-1]
# p = LDPortfolio(portf_scores, rating_type='SCORE', pd_cond_old=pd)
# vdb = VDBCalibration(portf_scores, rating_type='SCORE', target_ar=0.5,
#                      target_ct=0.15, pd_cond_old=pd, pd_uncond_target_ar=0.1)
# print(vdb.k)
# print(vdb.implied_ar)
# print(vdb.current_ct)
# print(vdb.pd_cond)

