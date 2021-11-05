
import numpy as np
import copy

class PSO:

    def __init__(self, l_bounds, u_bounds, dimensions):
        self.l_bounds = l_bounds
        self.u_bounds = u_bounds
        self.dimensions = dimensions
        self.pop_size = 5
        self.iteration = 0
        self.x = np.random.uniform(low=0.0, high=1.0, size=(self.pop_size, self.dimensions))
        self.v = np.random.randn(self.pop_size, dimensions)
        self.y = self.x.copy()
        self.pg = self.x[self.pop_size - 1, :].copy()
        self.f_pg = None
        self.p = np.zeros(self.pop_size)
        self.fx = np.zeros(self.pop_size)

        self.w = 0.5
        self.c1 = 2
        self.c2 = 2

    def parse_suggestions_history(self, suggestions_history):
        suggestions_history_use = copy.deepcopy(suggestions_history)
        x_datas = [[item[1] for item in sorted(suggestion[0].items(), key=lambda x: x[0])]
                   for suggestion in suggestions_history_use]
        y_datas = [suggestion[1] for suggestion in suggestions_history_use]
        x_datas = np.array(x_datas)
        y_datas = np.array(y_datas)
        # scale (data - lb) / (ub - lb) * (ub - lb) + lb
        # x_datas -= self.l_bounds
        # x_datas /= (self.u_bounds - self.l_bounds)
        # x_datas = (x_datas - self.l_bounds) / (self.u_bounds - self.l_bounds)
        return x_datas, y_datas

    def set_init(self, suggestions_history):
        # get 5 configs to use for warmstart
        x_datas, y_datas = self.parse_suggestions_history(suggestions_history)
        # 使用随机策略
        # x_max = x_datas[y_datas.argmax()]
        # max_y = y_datas.max()
        # rand_idx = np.random.choice(np.arange(len(suggestions_history)), self.pop_size - 1, replace=False)
        # x_rand = x_datas[rand_idx]
        # self.x[self.pop_size-1, :] = x_max
        # self.x[0:self.pop_size-1, :]= x_rand
        # self.fx[self.pop_size-1] = -max_y
        # self.fx[0:self.pop_size-1:] = -y_datas[rand_idx]
        # self.p = self.fx.copy()
        # self.y = self.x.copy()
        # self.pg = self.x[self.pop_size - 1, :].copy()
        # self.f_pg = self.p[self.pop_size - 1].copy()
        # 使用topk策略
        top_idx = np.argsort(y_datas)[-5:]
        self.x = x_datas[top_idx].astype(np.float)
        self.fx = -y_datas[top_idx]
        self.p = -y_datas[top_idx]
        self.y = self.x.copy()
        self.pg = self.x[self.pop_size - 1, :].copy()
        self.f_pg = self.p[self.pop_size - 1].copy()

        # print(self.population)
        # print(self.fitness)
        self.iteration = 1

    def boundary_check(self, vector, fix_type='random'):

        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector
        if fix_type == 'random':
            vector[violations] = np.random.uniform(low=0.0, high=1.0, size=len(violations))
        else:
            vector[violations] = np.clip(vector[violations], a_min=0, a_max=1)
        return vector

    def suggest(self, suggestions_history):
        if self.iteration == 0:
            self.set_init(suggestions_history)
        else:
            self.observe(suggestions_history)

        # print(self.fx)
        # print(self.p)
        # print(self.y)

        trials = []
        for i in range(self.pop_size):
            self.v[i, :] = self.w * self.v[i, :] + self.c1 * np.random.rand() * (
                    self.y[i, :] - self.x[i, :]) + self.c2 * np.random.rand() * (self.pg - self.x[i, :])
            self.x[i, :] = self.x[i, :] + self.v[i, :]
            self.x[i, :] = np.clip(self.x[i, :], self.l_bounds, self.u_bounds)
            # self.x[i, :] = self.boundary_check(self.x[i, :])
            trials.append(self.x[i, :])
        trials = np.array(trials)

        # inverse_transform
        # trials *= (self.u_bounds - self.l_bounds)
        # trials += self.l_bounds
        return trials

    def observe(self, suggestions_history):
        x_datas, y_datas = self.parse_suggestions_history(suggestions_history)
        y = y_datas[-5:]
        for i in range(self.pop_size):
            self.fx[i] = -y[i]
            if -y[i] < self.p[i]:
                self.p[i] = -y[i]
                self.y[i, :] = self.x[i, :].copy()
            if self.p[i] < self.f_pg:
                self.pg = self.y[i, :].copy()
                self.f_pg = self.p[i]

        self.iteration += 1


