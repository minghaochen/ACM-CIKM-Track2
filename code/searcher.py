# coding=utf-8
import copy
import random
import warnings

import numpy as np
# random.seed(42)
# np.random.seed(42)
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from scipy.stats import norm

# Need to import the searcher abstract class, the following are essential
from thpo.abstract_searcher import AbstractSearcher

from pyDOE2 import lhs
import GPy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import power_transform
import pandas as pd
from PSO_optimization import PSO

# fw = open("res.txt", "w", encoding='utf-8')

class GPyGP:

    def __init__(self):
        self.xscaler = MinMaxScaler((0, 1))
        self.yscaler = StandardScaler()
        self.num_epochs = 20
        self.verbose = False

    def fit_scaler(self, X, y):
        self.xscaler.fit(X)
        self.yscaler.fit(y)

    def trans(self, X, y=None):
        X = self.xscaler.transform(X)
        if y is not None:
            y = self.yscaler.transform(y)
            return X, y
        return X

    def fit(self, X, y):
        self.fit_scaler(X, y)
        X, y = self.trans(X, y)

        k1 = GPy.kern.Linear(X.shape[1], ARD=False)
        k2 = GPy.kern.Matern32(X.shape[1], ARD=True)
        k2.lengthscale = np.std(X, axis=0).clip(min=0.02)
        k2.variance = 0.5
        k2.variance.set_prior(GPy.priors.Gamma(0.5, 1), warning=False)
        kern = k1 + k2
        xmin = np.zeros(X.shape[1])
        xmax = np.ones(X.shape[1])
        warp_f = GPy.util.input_warping_functions.KumarWarping(X, Xmin=xmin, Xmax=xmax)
        self.gp = GPy.models.InputWarpedGP(X, y, kern, warping_function=warp_f)
        self.gp.likelihood.variance.set_prior(GPy.priors.LogGaussian(-4.63, 0.5), warning=False)
        # self.gp.optimize_restarts(max_iters=self.num_epochs, verbose=self.verbose, num_restarts=5, robust=True)
        self.gp.optimize(max_iters=self.num_epochs)


    def predict(self, X):
        X = self.trans(X)
        py, ps2 = self.gp.predict(X)
        mu = self.yscaler.inverse_transform(py.reshape(-1, 1))
        var = np.clip(self.yscaler.var_ * ps2.reshape(-1, 1), 1e-6, np.inf)
        return mu, var

    @property
    def noise(self):
        var_normalized = self.gp.likelihood.variance[0]
        noise = (var_normalized * self.yscaler.var_).reshape(1, )
        return noise

def take_res(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or abs(value - array[idx-1]) < abs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]


def my_AVERAGE_main(data_list):
    if len(data_list) == 0:
        return 0
    if len(data_list) > 2:
        data_list.remove(min(data_list))
        data_list.remove(max(data_list))
        average_data = float(sum(data_list)) / len(data_list)
        return average_data
    elif len(data_list) <= 2:
        average_data = float(sum(data_list)) / len(data_list)
        return average_data

class UtilityFunction(object):
    """
    This class mainly implements the collection function
    """
    def __init__(self, kind, kappa, x_i, ratio):
        self.kappa = kappa
        self.x_i = x_i

        self._iters_counter = 0
        self.ratio = ratio

        if kind not in ['ucb', 'ei', 'poi','myfun','MACE']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x_x, g_p, y_max):
        if self.kind == 'ucb':
            return self._ucb(x_x, g_p, self.kappa)
        if self.kind == 'ei':
            return self._ei(x_x, g_p, y_max, self.x_i)
        if self.kind == 'poi':
            return self._poi(x_x, g_p, y_max, self.x_i)
        if self.kind == 'myfun':
            return self._myfun(x_x, g_p, y_max, self.x_i, self.kappa, self.ratio)
        if self.kind == 'MACE':
            return self._MACE(x_x, g_p, y_max, self.x_i, self.kappa, self.ratio)

    @staticmethod
    def _ucb(x_x, g_p, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            py, ps2 = g_p.predict(x_x)
        py = py.reshape(-1, )
        ps2 = ps2.reshape(-1, )  # var or std
        return py + kappa * ps2

    @staticmethod
    def _ei(x_x, g_p, y_max, x_i):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = g_p.predict(x_x)

        mean = mean.reshape(-1, )
        std = std.reshape(-1, )  # var or std
        std = np.sqrt(std)
        a_a = (mean - y_max - x_i)
        z_z = a_a / std
        return a_a * norm.cdf(z_z) + std * norm.pdf(z_z)

    @staticmethod
    def _poi(x_x, g_p, y_max, x_i):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = g_p.predict(x_x)

        mean = mean.reshape(-1, )
        std = std.reshape(-1, )  # var or std
        std = np.sqrt(std)
        z_z = (mean - y_max - x_i)/std
        return norm.cdf(z_z)

    @staticmethod
    def _myfun(x_x, g_p, y_max, x_i, kappa, ratio):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = g_p.predict(x_x)

        mean = mean.reshape(-1,)
        std = std.reshape(-1,) # var or std

        y1 = mean + kappa * std
        a_a = (mean - y_max - x_i)
        z_z = a_a / std
        y2 = a_a * norm.cdf(z_z) + std * norm.pdf(z_z)
        z_z = (mean - y_max - x_i) / std
        y3 = norm.cdf(z_z)
        return ratio[0]*y1 + ratio[1]*y2 + ratio[2]*y3

    @staticmethod
    def _MACE(x_x, g_p, y_max, x_i, kappa, ratio):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            py, ps2 = g_p.predict(x_x)

        eps = 1e-4

        py = py.reshape(-1, )
        ps2 = ps2.reshape(-1, )  # var or std
        noise = np.sqrt(2.0 * g_p.noise)
        ps = np.sqrt(ps2)
        ucb = (py + noise * np.random.randn(*py.shape)) + kappa * ps

        a_a = (py + noise * np.random.randn(*py.shape) - y_max - x_i)
        z_z = a_a / ps
        EI = a_a * norm.cdf(z_z) + ps * norm.pdf(z_z)

        z_z = (py + noise * np.random.randn(*py.shape) - y_max - x_i) / ps
        PI = norm.cdf(z_z)

        DI = (py + noise * np.random.randn(*py.shape) - y_max - x_i) / a_a
        DI = a_a * norm.cdf(DI) + ps * norm.pdf(DI)

        return ratio[0]*ucb + ratio[1]*EI + ratio[2]*PI + ratio[3]*DI


class Searcher(AbstractSearcher):

    def __init__(self, parameters_config, n_iter, n_suggestion):
        """ Init searcher

        Args:
            parameters_config: parameters configuration, consistent with the definition of parameters_config of EvaluateFunction. dict type:
                    dict key: parameters name, string type
                    dict value: parameters configuration, dict type:
                        "parameter_name": parameter name
                        "parameter_type": parameter type, 1 for double type, and only double type is valid
                        "double_max_value": max value of this parameter
                        "double_min_value": min value of this parameter
                        "double_step": step size
                        "coords": list type, all valid values of this parameter.
                            If the parameter value is not in coords,
                            the closest valid value will be used by the judge program.

                    parameter configuration example, eg:
                    {
                        "p1": {
                            "parameter_name": "p1",
                            "parameter_type": 1
                            "double_max_value": 2.5,
                            "double_min_value": 0.0,
                            "double_step": 1.0,
                            "coords": [0.0, 1.0, 2.0, 2.5]
                        },
                        "p2": {
                            "parameter_name": "p2",
                            "parameter_type": 1,
                            "double_max_value": 2.0,
                            "double_min_value": 0.0,
                            "double_step": 1.0,
                            "coords": [0.0, 1.0, 2.0]
                        }
                    }
                    In this example, "2.5" is the upper bound of parameter "p1", and it's also a valid value.

        n_iteration: number of iterations
        n_suggestion: number of suggestions to return
        """
        AbstractSearcher.__init__(self, parameters_config, n_iter, n_suggestion)

        self.gp = GPyGP()
        l_bounds = []
        u_bounds = []
        for p_name, p_conf in sorted(self.parameters_config.items(), key=lambda x: x[0]):
            l_bounds.append(p_conf['double_min_value'])
            u_bounds.append(p_conf['double_max_value'])
        self.l_bounds = l_bounds
        self.u_bounds = u_bounds

        samp = lhs(len(self.parameters_config), 10, criterion='maximin', iterations=100, random_state=2021)
        # samp = lhs(len(self.parameters_config), 10)
        self.samp = samp * (np.array(u_bounds) - np.array(l_bounds)) + np.array(l_bounds)
        self.PSO = PSO(self.l_bounds, self.u_bounds, len(self.parameters_config))


    def init_param_group(self, n_suggestions):
        """ Suggest n_suggestions parameters in random form

        Args:
            n_suggestions: number of parameters to suggest in every iteration

        Return:
            next_suggestions: n_suggestions Parameters in random form
        """
        next_suggestions = [{p_name: p_conf['coords'][random.randint(0, len(p_conf["coords"]) - 1)]
                            for p_name, p_conf in self.parameters_config.items()} for _ in range(n_suggestions)]

        return next_suggestions

    def parse_suggestions_history(self, suggestions_history):
        """ Parse historical suggestions to the form of (x_datas, y_datas), to obtain GP training data

        Args:
            suggestions_history: suggestions history

        Return:
            x_datas: Parameters
            y_datas: Reward of Parameters
        """
        suggestions_history_use = copy.deepcopy(suggestions_history)
        x_datas = [[item[1] for item in sorted(suggestion[0].items(), key=lambda x: x[0])]
                   for suggestion in suggestions_history_use]
        y_datas = [suggestion[1] for suggestion in suggestions_history_use]
        x_datas = np.array(x_datas)
        y_datas = np.array(y_datas)
        return x_datas, y_datas

    def train_gp(self, x_datas, y_datas):
        """ train gp

        Args:
            x_datas: Parameters
            y_datas: Reward of Parameters

        Return:
            gp: Gaussian process regression
        """
        try:
            self.gp.fit(x_datas, y_datas)
        except:
            y_datas = y_datas.reshape(-1, 1)
            self.gp.fit(x_datas, y_datas)

    def random_sample(self):
        """ Generate a random sample in the form of [value_0, value_1,... ]

        Return:
            sample: a random sample in the form of [value_0, value_1,... ]
        """
        sample = [p_conf['coords'][random.randint(0, len(p_conf["coords"]) - 1)] for p_name, p_conf
                  in sorted(self.parameters_config.items(), key=lambda x: x[0])]
        return sample

    def lhs_sample(self, num_sample):

        samp = lhs(len(self.parameters_config), num_sample)
        samp = samp * (np.array(self.u_bounds) - np.array(self.l_bounds)) + np.array(self.l_bounds)
        return samp

    def get_bounds(self):
        """ Get sorted parameter space

        Return:
            _bounds: The sorted parameter space
        """
        def _get_param_value(param):
            value = [param['double_min_value'], param['double_max_value']]
            return value

        _bounds = np.array(
            [_get_param_value(item[1]) for item in sorted(self.parameters_config.items(), key=lambda x: x[0])],
            dtype=np.float
        )
        return _bounds

    def acq_max(self, f_acq, gp, y_max, bounds, num_warmup, num_starting_points):
        """ Produces the best suggested parameters

        Args:
            f_acq: Acquisition function
            gp: GaussianProcessRegressor
            y_max: Best reward in suggestions history
            bounds: The parameter boundary of the acquisition function
            num_warmup: The number of samples randomly generated for the collection function
            num_starting_points: The number of random samples generated for scipy.minimize

        Return:
            Return the current optimal parameters
        """
        # Warm up with random points
        x_tries = np.array([self.random_sample() for _ in range(int(num_warmup))])
        ys = f_acq(x_tries, g_p=gp, y_max=y_max)
        x_max = x_tries[ys.argmax()]
        max_acq = ys.max()
        # Explore the parameter space more throughly
        x_seeds = np.array([self.random_sample() for _ in range(int(num_starting_points))])
        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            res = minimize(lambda x: -f_acq(x.reshape(1, -1), g_p=gp, y_max=y_max),
                           x_try.reshape(1, -1),
                           bounds=bounds,
                           method="L-BFGS-B")
            # See if success
            if not res.success:
                continue
            # Store it if better than previous minimum(maximum).
            if max_acq is None or -res.fun[0] >= max_acq:
                x_max = res.x
                max_acq = -res.fun[0]
        return np.clip(x_max, bounds[:, 0], bounds[:, 1])

    def parse_suggestions(self, suggestions):
        """ Parse the parameters result

        Args:
            suggestions: Parameters

        Return:
            suggestions: The parsed parameters
        """
        def get_param_value(p_name, value):
            p_coords = self.parameters_config[p_name]['coords']
            if value in p_coords:
                return value
            else:
                subtract = np.abs([p_coord - value for p_coord in p_coords])
                min_index = np.argmin(subtract, axis=0)
                return p_coords[min_index]

        p_names = [p_name for p_name, p_conf in sorted(self.parameters_config.items(), key=lambda x: x[0])]
        suggestions = [{p_names[index]:suggestion[index] for index in range(len(suggestion))}
                       for suggestion in suggestions]

        suggestions = [{p_name: get_param_value(p_name, value) for p_name, value in suggestion.items()}
                       for suggestion in suggestions]
        return suggestions

    def check_unique(self, X, rec):
        X = pd.DataFrame(X)
        rec = pd.DataFrame(rec)
        return (~pd.concat([X, rec], axis = 0).duplicated().tail(rec.shape[0]).values).tolist()


    def suggest_old(self, suggestions_history, n_suggestions=1):
        """ Suggest next n_suggestion parameters, old implementation of preliminary competition.

        Args:
            suggestions_history: a list of historical suggestion parameters and rewards, in the form of
                    [[Parameter, Reward], [Parameter, Reward] ... ]
                        Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                            {'p1': 0, 'p2': 0, 'p3': 0}
                        Reward: a float type value

                    The parameters and rewards of each iteration are placed in suggestions_history in the order of iteration.
                        len(suggestions_history) = n_suggestion * iteration(current number of iteration)

                    For example:
                        when iteration = 2, n_suggestion = 2, then
                        [[{'p1': 0, 'p2': 0, 'p3': 0}, -222.90621774147272],
                         [{'p1': 0, 'p2': 1, 'p3': 3}, -65.26678723205647],
                         [{'p1': 2, 'p2': 2, 'p3': 2}, 0.0],
                         [{'p1': 0, 'p2': 0, 'p3': 4}, -105.8151893979122]]

            n_suggestion: int, number of suggestions to return

        Returns:
            next_suggestions: list of Parameter, in the form of
                    [Parameter, Parameter, Parameter ...]
                        Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                            {'p1': 0, 'p2': 0, 'p3': 0}

                    For example:
                        when n_suggestion = 3, then
                        [{'p1': 0, 'p2': 0, 'p3': 0},
                         {'p1': 0, 'p2': 1, 'p3': 3},
                         {'p1': 2, 'p2': 2, 'p3': 2}]
        """
        if (suggestions_history is None) or (len(suggestions_history) <= 5):

            iter = len(suggestions_history) // 5
            suggest_val = self.samp[iter * n_suggestions:(iter + 1) * n_suggestions, :]
            next_suggestions = []
            for k in range(n_suggestions):
                one_suggest = {}
                count = 0
                for p_name, p_conf in sorted(self.parameters_config.items(), key=lambda x: x[0]):
                    one_suggest[p_name] = take_res(p_conf["coords"], suggest_val[k, count])
                    count += 1
                next_suggestions.append(one_suggest)

        # elif current_time - self.start < 610.0 - 0*(self.epoch_end - self.epoch_start):
        elif len(suggestions_history) <= 50:

            x_datas, y_datas = self.parse_suggestions_history(suggestions_history)
            try:
                if min(y_datas) <= 0:
                    y = power_transform(y_datas.reshape(-1, 1) / np.std(y_datas), method='yeo-johnson')
                else:
                    y = power_transform(y_datas.reshape(-1, 1) / np.std(y_datas), method='box-cox')
                    if np.std(y) < 0.5:
                        y = power_transform(y / np.std(y), method='yeo-johnson')
                if np.std(y) < 0.5:
                    raise RuntimeError('Power transformation failed')
                self.train_gp(x_datas, y)
            except:
                y = y_datas.copy()
                self.train_gp(x_datas, y)

            # print("max",max(y))
            # print("min", min(y))

            _bounds = self.get_bounds()
            suggestions = []

            upsi = 0.5
            delta = 0.01
            iter = len(suggestions_history) // 5
            kappa = np.sqrt(
                upsi * 2 * ((2.0 + x_datas.shape[1] / 2.0) * np.log(iter) + np.log(3 * np.pi ** 2 / (3 * delta))))

            Ratio = lhs(4, n_suggestions)

            for index in range(n_suggestions):
                # ratio = np.random.rand(3)
                # ratio /= sum(ratio)

                ratio = Ratio[index, :]
                ratio /= sum(ratio)

                # kappa = 2.576 * (random.randint(0, 4) + 1)
                # x_i = 5 * random.randint(0, 4)
                # kappa = 2.576
                x_i = 1e-4
                x_i = (max(y) - min(y)) / 100 * 10
                x_i = abs(max(y)) * 0.05
                if len(suggestions_history) % 5 == 0:
                    gain = 4
                elif len(suggestions_history) % 4 == 0:
                    gain = 3
                elif len(suggestions_history) % 3 == 0:
                    gain = 2
                elif len(suggestions_history) % 2 == 0:
                    gain = 1
                else:
                    gain = 0
                if max(y) > 0:
                    x_i = abs(max(y)) * 0.05 * gain + 1e-4# 考虑负数 max(abs()) random.randint(0,4)
                else:
                    x_i = abs(min(y)) * 0.05 * gain + 1e-4
                # x_i = max(abs(y)) * 0.1

                utility_function = UtilityFunction(kind='MACE', kappa=kappa,
                                                   x_i=x_i, ratio=ratio)
                # utility_function = UtilityFunction(kind='MACE', kappa=kappa,
                #                                    x_i= kappa / 100 , ratio=ratio)
                # print("ratip", ratio, "kappa", kappa, "x_i", x_i)

                suggestion = self.acq_max(
                    f_acq=utility_function.utility,
                    gp=self.gp,
                    # y_max=y_datas.max(),
                    y_max=y.max(),
                    bounds=_bounds,
                    num_warmup=5000,
                    num_starting_points=1,
                    # num_starting_points=5
                )
                suggestions.append(suggestion)

            suggestions = np.array(suggestions)
            # print(suggestions)
            # 去重
            suggestions = suggestions[self.check_unique(x_datas, suggestions)].reshape(-1, len(self.parameters_config))
            if suggestions.shape[0] < n_suggestions:
                # print('去重')
                samp = self.lhs_sample(n_suggestions - suggestions.shape[0])
                suggestions = np.vstack([suggestions, samp])

            suggestions = self.parse_suggestions(suggestions)
            next_suggestions = suggestions

        else:
            # 启发式
            # suggestions = self.DE.suggest(suggestions_history)
            suggestions = self.PSO.suggest(suggestions_history)

            x_datas, y_datas = self.parse_suggestions_history(suggestions_history)
            suggestions = suggestions[self.check_unique(x_datas, suggestions)].reshape(-1, len(self.parameters_config))
            if suggestions.shape[0] < n_suggestions:
                # print('去重')
                samp = self.lhs_sample(n_suggestions - suggestions.shape[0])
                suggestions = np.vstack([suggestions, samp])

            suggestions = self.parse_suggestions(suggestions)
            next_suggestions = suggestions


        return next_suggestions

    def get_my_score(self, reward):
        """ Get the most trusted reward of all iterations.

        Returns:
            most_trusted_reward: float
        """
        res = []
        for reward_one in reward:
            res.append(reward_one['value'])
        # return np.mean(res)
        return min(res)
        # return reward[-1]['value']


    def suggest(self, iteration_number, running_suggestions, suggestion_history, n_suggestions=1):
        """ Suggest next n_suggestion parameters. new implementation of final competition

        Args:
            iteration_number: int ,the iteration number of experiment, range in [1, 140]

            running_suggestions: a list of historical suggestion parameters and rewards, in the form of
                    [{"parameter": Parameter, "reward": Reward}, {"parameter": Parameter, "reward": Reward} ... ]
                Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                    {'p1': 0, 'p2': 0, 'p3': 0}
                Reward: a list of dict, each dict of the list corresponds to an iteration,
                    the dict is in the form of {'value':value,  'upper_bound':upper_bound, 'lower_bound':lower_bound}
                    Reward example:
                        [{'value':1, 'upper_bound':2,   'lower_bound':0},   # iter 1
                         {'value':1, 'upper_bound':1.5, 'lower_bound':0.5}  # iter 2
                        ]

            suggestion_history: a list of historical suggestion parameters and rewards, in the same form of running_suggestions

            n_suggestion: int, number of suggestions to return

        Returns:
            next_suggestions: list of Parameter, in the form of
                    [Parameter, Parameter, Parameter ...]
                        Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                            {'p1': 0, 'p2': 0, 'p3': 0}
                    For example:
                        when n_suggestion = 3, then
                        [{'p1': 0, 'p2': 0, 'p3': 0},
                         {'p1': 0, 'p2': 1, 'p3': 3},
                         {'p1': 2, 'p2': 2, 'p3': 2}]
        """
        # MIN_TRUSTED_ITERATION = 8
        # new_suggestions_history = []
        # for suggestion in suggestion_history:
        #     iterations_of_suggestion = len(suggestion['reward'])
        #     if iterations_of_suggestion >= MIN_TRUSTED_ITERATION:
        #         cur_score = self.get_my_score(suggestion['reward'])
        #         new_suggestions_history.append([suggestion["parameter"], cur_score])
        # return self.suggest_old(new_suggestions_history, n_suggestions)

        new_suggestions_history = []
        for suggestion in suggestion_history:
            new_suggestions_history.append([suggestion["parameter"], suggestion['reward'][-1]['value']])
        return self.suggest_old(new_suggestions_history, n_suggestions)

    def is_early_stop(self, iteration_number, running_suggestions, suggestion_history):
        """ Decide whether to stop the running suggested parameter experiment.

        Args:
            iteration_number: int, the iteration number of experiment, range in [1, 140]

            running_suggestions: a list of historical suggestion parameters and rewards, in the form of
                    [{"parameter": Parameter, "reward": Reward}, {"parameter": Parameter, "reward": Reward} ... ]
                Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                    {'p1': 0, 'p2': 0, 'p3': 0}
                Reward: a list of dict, each dict of the list corresponds to an iteration,
                    the dict is in the form of {'value':value,  'upper_bound':upper_bound, 'lower_bound':lower_bound}
                    Reward example:
                        [{'value':1, 'upper_bound':2,   'lower_bound':0},   # iter 1
                         {'value':1, 'upper_bound':1.5, 'lower_bound':0.5}  # iter 2
                        ]

            suggestions_history: a list of historical suggestion parameters and rewards, in the same form of running_suggestions

        Returns:
            stop_list: list of bool, indicate whether to stop the running suggestions.
                    len(stop_list) must be the same as len(running_suggestions), for example:
                        len(running_suggestions) = 3, stop_list could be :
                            [True, True, True] , which means to stop all the three running suggestions
        """

        # Early Stop algorithm demo 2:
        #
        #   If there are 3 or more suggestions which had more than 7 iterations,
        #   the worst running suggestions will be stopped
        #

        # print(iteration_number)
        MIN_ITERS_TO_STOP = 8
        MIN_SUGGUEST_COUNT_TO_STOP = 2
        MAX_ITERS_OF_DATASET = self.n_iteration
        ITERS_TO_GET_STABLE_RESULT = 14
        INITIAL_INDEX = -1

        res = [False] * len(running_suggestions)
        if iteration_number + ITERS_TO_GET_STABLE_RESULT <= MAX_ITERS_OF_DATASET:
            score_min_idx = INITIAL_INDEX
            score_min = float("inf")
            count = 0
            # Get the worst suggestion of current running suggestions
            for idx, suggestion in enumerate(running_suggestions):
                if len(suggestion['reward']) >= MIN_ITERS_TO_STOP:
                    count = count + 1
                    cur_score = self.get_my_score(suggestion['reward'])
                    if score_min_idx == INITIAL_INDEX or cur_score < score_min:
                        score_min_idx = idx
                        score_min = cur_score
            # Stop the worst suggestion
            if count >= MIN_SUGGUEST_COUNT_TO_STOP and score_min_idx != INITIAL_INDEX:
                res[score_min_idx] = True
        res = [False] * len(running_suggestions)
        # if iteration_number < 100 and iteration_number%7==0:
        #     res =  [True] * len(running_suggestions)
        return res

    # 早停
    # def suggest(self, iteration_number, running_suggestions, suggestion_history, n_suggestions=1):
    #     if len(suggestion_history) >= 10:
    #         self.updatehis(suggestion_history)
    #         fw.write("最优值上下界")
    #         fw.write('\n')
    #         fw.write("=".join(map(str, self.hisreward)))
    #         fw.write('\n')
    #         fw.write("当前最优值")
    #         fw.write('\n')
    #         fw.write(str(self.maxreward))
    #         fw.write('\n')
    #         print("最优值的上下界", self.hisreward)
    #         print("当前最优值:", self.maxreward)
    #     new_suggestions_history = []
    #
    #     for suggestion in suggestion_history:
    #         if len(suggestion['reward']) == 14:
    #             tem = [suggestion['reward'][i]['value'] for i in range(len(suggestion['reward']))]
    #             new_suggestions_history.append([suggestion["parameter"], suggestion['reward'][-1]['value']])
    #             fw.write("跑完的数据")
    #             fw.write('\n')
    #             fw.write("=".join(map(str, tem)))
    #             fw.write('\n')
    #         else:
    #             tem = [suggestion['reward'][i]['value'] for i in range(len(suggestion['reward']))]
    #             print("未跑完的长度:", tem)
    #             fw.write("未跑完的长度" + "\n")
    #             fw.write("=".join(map(str, tem)) + "\n")
    #             if len(tem) >= 7:
    #                 tem = tem[-7:]
    #                 tem.remove(max(tem))
    #                 tem.remove(min(tem))
    #                 if max(tem) - min(tem) <= 1.2 * (max(self.hisreward) - min(self.hisreward)):
    #                     new_suggestions_history.append(
    #                         [suggestion["parameter"], min(tem)])
    #                     fw.write("次优值的上下界" + "\t")
    #                     fw.write("=".join(map(str, tem)) + "\t")
    #
    #                     print("次优值的上下界", tem)
    #     print("训练数据的数量：", len(new_suggestions_history))
    #     return self.suggest_old(new_suggestions_history, n_suggestions)
    #
    # def updatehis(self, suggestion_history):
    #     self.result5 = []
    #     self.result10 = []
    #
    #     self.hisreward = []
    #     self.maxreward = float("-inf")
    #
    #     for suggestion in suggestion_history:
    #         if len(suggestion['reward']) == 14:
    #             if suggestion['reward'][-1]['value'] > self.maxreward:
    #                 self.maxreward = suggestion['reward'][-1]['value']
    #                 self.hisreward = [suggestion['reward'][i]['value'] for i in range(13)]
    #     self.hisreward.remove(max(self.hisreward))
    #     self.hisreward.remove(min(self.hisreward))
    #     self.hisreward = self.hisreward + [self.maxreward]
    #
    # def is_early_stop(self, iteration_number, running_suggestions, suggestion_history):
    #
    #     res = [False] * len(running_suggestions)
    #     if len(suggestion_history) < 25:  # 在初始化的时候直接跑满
    #         res = [False] * len(running_suggestions)
    #         return res
    #     else:
    #         self.updatehis(suggestion_history)
    #         for ind, suggestion in enumerate(running_suggestions):
    #             tem = [suggestion['reward'][i]['value'] for i in range(len(suggestion['reward']))]
    #             fw.write("=".join(map(str, tem)) + "\n")
    #             if len(suggestion) >= 3 and suggestion['reward'][-1]['upper_bound'] < self.maxreward:
    #                 res[ind] = True
    #                 fw.write("上界去除" + '\n')
    #             tem = [suggestion['reward'][i]['value'] for i in range(len(suggestion['reward']))]
    #             ##如果它之前value的最大值小于的话那也很难
    #             if len(suggestion['reward']) >= 7 and max(tem[-7:]) < self.maxreward:
    #                 res[ind] = True
    #                 fw.write("最后5次去除" + '\n')
    #
    #             if len(tem) >= 7:
    #                 tem.remove(max(tem))
    #                 tem.remove(min(tem))
    #                 if max(tem) - min(tem) > 1.25 * (max(self.hisreward) - min(self.hisreward)):
    #                     fw.write("上下界去除" + '\n')
    #                     res[ind] = True
    #
    #             if 140 - iteration_number < 5:
    #                 res[ind] = False
    #         print("返回的结果：", res)
    #     return res