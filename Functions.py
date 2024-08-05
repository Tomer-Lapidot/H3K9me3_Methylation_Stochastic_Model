import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import mixture
import itertools
import scipy
from scipy.optimize import curve_fit


class Model():
    def __init__(self, _mod_params):

        self._kp = _mod_params['k+']
        self._kn = _mod_params['k-']
        self._tf = _mod_params['tf']
        self._dt = _mod_params['dt']
        self._n = _mod_params['n']

    def run_model(self, initial='None'):

        _time = np.arange(0, self._tf, self._dt)
        _data = np.zeros([len(_time), self._n])

        # The model

        if isinstance(initial, str):
            _nucleosomes = np.zeros([self._n])
        else:
            _nucleosomes = initial.copy()

        _edge = False
        for i in range(len(_time)):  # Time loop, index i
            for j in range(self._n):  # Space (nucleosome position) loop, index j

                # Nucleation process
                if j == int(self._n / 2):
                    if _nucleosomes[j] == 0:
                        _nucleosomes[j] = Model.nucleate(self._kp, self._dt, _nucleosomes[j])

                # Propegration processes
                if j == 0:
                    if _nucleosomes[j] == 1 and _nucleosomes[j + 1] == 0:
                        _nucleosomes[j + 1] = Model.propagation(self._kp, self._dt, _nucleosomes[j + 1])

                if j > 0 and j < self._n - 1:
                    if _nucleosomes[j] == 1 and _nucleosomes[j - 1] == 0:
                        _nucleosomes[j - 1] = Model.propagation(self._kp, self._dt, _nucleosomes[j - 1])

                    if _nucleosomes[j] == 1 and _nucleosomes[j + 1] == 0:
                        _nucleosomes[j + 1] = Model.propagation(self._kp, self._dt, _nucleosomes[j + 1])

                if j == self._n - 1:
                    if _nucleosomes[j] == 1 and _nucleosomes[j - 1] == 0:
                        _nucleosomes[j - 1] = Model.propagation(self._kp, self._dt, _nucleosomes[j - 1])

                # Turnover process
                if _nucleosomes[j] == 1:
                    _nucleosomes[j] = Model.turnover(self._kn, self._dt, _nucleosomes[j])

                # Record model output, for time i and position j
                _data[i, j] = _nucleosomes[j]

        if _data[:, 0].sum() > 0 or _data[:, -1].sum() > 0:
            _edge = True

        return Data(_data, _time, _nucleosomes, _edge)

    @staticmethod
    def nucleate(_kp, _dt, _a):

        p = np.random.random()

        if _kp*_dt > p:
            _a = 1

        return _a

    @staticmethod
    def propagation(_kp, _t, _a):

        p = np.random.random()

        if _kp*_t > p:
            _a = 1

        return _a

    @staticmethod
    def turnover(_kn, _t, _a):

        p = np.random.random()

        if _kn*_t > p:
            _a = 0

        return _a

    def Plot_Model_Results(self, _output):

        k = 0
        points = np.zeros([int(_output.sum()), 2])
        for i in range(len(_output[:, 0])):
            for j in range(len(_output[0, :])):
                if _output[i, j] == 1:
                    points[k, 0] = i
                    points[k, 1] = j
                    k = k + 1

        plt.scatter(points[:, 1], len(_output) - points[:, 0], s=1, c='black')

        plt.show()


class Data():
    def __init__(self, _data, _time, _space, _edge):

        self._data = _data
        self._time = _time
        self._space = _space
        self._edge = _edge
        self.fit_params = np.array([None, None, None])

    def len(self):
            return len(self._data)

    def shape(self):
            return self._data.shape

    def to_numpy(self):
            return self._data

    def get_edge(self):
            return self._edge

    def final_time(self):
            return self._data[-1]

    def get_time(self):
            return self._time

    def get_exp_time(self, _exp_time):

            exp_time_out = np.zeros([len(_exp_time), 2])
            for i, t in enumerate(_exp_time):
                idx = np.squeeze(np.where(self._time == t)[0])
                exp_time_out[i, 0] = self._time[idx]
                exp_time_out[i, 1] = Data.time_clusters(self)[idx]

            return pd.DataFrame(exp_time_out, columns=['time', 'nucleosomes'])

    def get_space(self):
            return self._space

    def plot_data(self):
            plt.imshow(self._data, cmap='Greys', aspect='auto')
            plt.xlabel('Nucleosome Space')
            plt.ylabel('Time')

    def plot_nucleosomes(self, color='black'):
        plt.plot(self._time, Data.time_clusters(self), color=color)
        plt.xlabel('Time')
        plt.ylabel('Total Active Nucleosomes')

    def time_clusters(self):
        return self._data.sum(axis=1)

    def fit_exp(self):

        _t = self._time
        _y = Data.time_clusters(self)

        self.fit_params = regress_line(_t, _y)

    def get_params(self):
        return self.fit_params

    def plot_fit(self, color='black', show_title=True):

        plt.scatter(self._time, Data.time_clusters(self), color=color, s=3)

        plt.xlabel('Time')
        plt.ylabel('Total Active Nucleosomes')

        time_mod = np.arange(0, self._time.max(), self._time.max()/1000)
        if np.any(self.fit_params) == None:
            print('Plot Warning: Data as not been fitted yet, no model curve exists')
        else:
            if show_title:
                plt.title('r2 = ' + str(np.round(self.fit_params[2], 4)))
            plt.plot(time_mod, self.fit_params[0]*np.exp(-self.fit_params[1]*time_mod), color=color)


class Plot:

    @staticmethod
    def plot_run(_model_run):
        title_name = ['Activation', 'Deactivation']
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.title(title_name[i])
            _model_run[i].plot_data()

    @staticmethod
    def plot_time_run(_model_run):
        title_name = ['Activation', 'Deactivation']
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.title(title_name[i])
            _model_run[i].plot_nucleosomes()

    @staticmethod
    def plot_final_time_model(_model_bank, color='black', color_ribon='lightgrey'):

        _, x = _model_bank[0, 0].shape()
        n = len(_model_bank)
        sum_array = np.zeros([n, x], dtype=int)

        for i in range(n):
            sum_array[i] = _model_bank[i, 0].final_time()

        x_mod = range(x)
        u_mod = np.mean(sum_array, axis=0)
        s_mod = np.std(sum_array, axis=0)

        plt.plot(x_mod, u_mod, color=color)
        plt.fill_between(x_mod, u_mod + s_mod, (u_mod - s_mod) * ((u_mod - s_mod) >= 0), color=color_ribon, alpha=0.33)

        plt.xlabel('Nucleosome Space')
        plt.ylabel('Frequency')

    @staticmethod
    def plot_gaussian_2D(_X, _Y, bins=20, contours=3, color='black'):

        def get_contour(_data_contour, _value):
            c_min = _data_contour.min()
            c_max = _data_contour.max()
            sweep = np.arange(c_min, c_max, (c_max - c_min) / 100)
            results = np.zeros([len(sweep), 2])

            total_vol = np.sum(_data_contour)
            for i, c in enumerate(sweep):
                results[i] = [c, np.sum(_data_contour * (_data_contour >= c)) / total_vol]

            idx = (np.abs(results[:, 1] - _value)).argmin()

            return results[idx, 0]

        data_raw, bx, by = np.histogram2d(_X, _Y, bins=bins)

        _x = bx[:-1] / 2 + bx[1:] / 2
        _y = by[:-1] / 2 + by[1:] / 2

        data = data_raw.T.ravel()
        _x, _y = np.meshgrid(_x, _y)

        initial_guess = (10, np.mean(_X), np.mean(_Y), 1, 1, 0)
        popt, pcov = curve_fit(gaussian_2D, (_x, _y), data, p0=initial_guess)

        # plt.imshow(data.reshape(len(_x), len(_y)), cmap=plt.cm.jet, origin='lower', extent=(_x.min(), _x.max(), _y.min(), _y.max()))
        # plt.scatter(_X, _Y, s=1, color=color)
        x_con = np.arange(_x.min(), _x.max(), (_x.max() - _x.min()) / 80)
        y_con = np.arange(_y.min(), _y.max(), (_y.max() - _y.min()) / 80)
        x_con_g, y_con_g = np.meshgrid(x_con, y_con)
        data_fitted = gaussian_2D((x_con_g, y_con_g), *popt)
        data_contour = data_fitted.reshape(len(x_con), len(y_con))

        max_idx = np.unravel_index(np.argmax(data_contour, axis=None), data_contour.shape)

        con_1 = get_contour(data_contour, 0.682)
        # con_2 = get_contour(data_contour, 0.954)
        # con_3 = get_contour(data_contour, 0.996)

        plt.contour(x_con_g, y_con_g, data_contour, [con_1], colors=color)
        plt.scatter(x_con[max_idx[1]], y_con[max_idx[0]], s=5, color=color)

    @staticmethod
    def plot_2D_hist(X, Y, colormap='jet'):

        H, ye, xe = np.histogram2d(Y, X, bins=1024)
        sH = scipy.ndimage.filters.gaussian_filter(H, sigma=10, order=0, mode='constant', cval=0.0)
        Hind = np.ravel(H)

        xc = (xe[:-1] + xe[1:]) / 2.0
        yc = (ye[:-1] + ye[1:]) / 2.0

        xv, yv = np.meshgrid(xc, yc)
        x_new = np.ravel(xv)[Hind != 0]
        y_new = np.ravel(yv)[Hind != 0]
        z_new = np.ravel(H if sH is None else sH)[Hind != 0]

        plt.scatter(x_new, y_new, c=z_new, s=1, cmap=colormap)

    @staticmethod
    def plot_2D_contours(_X, _Y, b=10, color_con='Greys_r', lim_low=0.2):

        _Z, _Yb, _Xb = np.histogram2d(_X, _Y, bins=(b, b))
        _Yb = _Yb[:-1] / 2 + _Yb[1:] / 2
        _Xb = _Xb[:-1] / 2 + _Xb[1:] / 2
        _XX, _YY = np.meshgrid(_Xb, _Yb, indexing='xy')

        # z_max = _Z.max()

        _Z = (_Z - _Z.min()) / (_Z.max() - _Z.min())

        plt.contour(_YY, _XX, _Z, cmap=color_con, linewidth=0, edgecolors='none', levels=np.arange(lim_low, 1, 0.1))


def regress_line(_t, _y, verbose=False):

    _t = _t[_y > 0]
    _y = _y[_y > 0]
    _y = np.log(_y)

    if len(_y) > 1:

        params, _ = curve_fit(lambda _t, _a, _b: _a - _b*_t, _t, _y)
        _y_mod =  params[0] - params[1] * _t
        _R2 = get_r2(_y, _y_mod)

        if verbose:
            plt.title('R2: ' + str(_R2))
            plt.scatter(_t, _y, color='black')
            plt.plot(_t, _y_mod, color='black')
            plt.show()

        return np.exp(params[0]), params[1], _R2

    else:
        if verbose:
            print('Error: One or less non-zero points, cannot fit line')

        return [None, None, None]

def get_r2(_y, _y_mod):
    corr_matrix = np.corrcoef(_y, _y_mod)
    corr = corr_matrix[0, 1]
    _r_squared = corr ** 2

    return _r_squared

def cycle_colors(_c, _n):

    # c: Number of colors through
    # n: Length of data

    _colors = plt.cm.jet(np.linspace(0, 1, _c))
    _colors_rep = _colors.copy()
    for i in range(int(np.ceil(_n / _c)) - 1):
        _colors_rep = np.concatenate((_colors_rep, _colors), axis=0)

    return _colors_rep

def clean_zeros_nans(_X, _Y):

    _X = _X[~np.isnan(_X)]
    _Y = _Y[~np.isnan(_Y)]

    _X = _X[_X > 0]
    _Y = _Y[_X > 0]

    _X = _X[_Y > 0]
    _Y = _Y[_Y > 0]

    return _X, _Y

def gaussian_2D(xy, amplitude, xo, yo, sigma_x, sigma_y, theta): #, offset):

    x, y = xy

    xo = float(xo)
    yo = float(yo)

    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)

    g = amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))

    return g.ravel()

def cluster_by_gaussian(_x, _y, n_cluster=2, seed=None, verbose=False):

    _X = np.concatenate((np.expand_dims(_x, -1), np.expand_dims(_y, -1)), axis=1)

    if seed != 'None':
        np.random.seed(seed)

    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, n_cluster + 1)
    cv_types = ["spherical", "tied", "diag", "full"]
    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
            gmm.fit(_X)
            bic.append(gmm.bic(_X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    color_iter = itertools.cycle(['black', 'purple', 'blue', 'green', 'gold', 'orange', 'red', 'pink', 'brown', 'grey'])
    clf = best_gmm

    _Y = clf.predict(_X)

    if verbose:
        # color_list = ['Blues_r', 'Greens_r', 'Reds_r']
        for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
            if not np.any(_Y == i):
                continue
            Plot.plot_2D_hist(_X[_Y == i, 0], _X[_Y == i, 1], colormap='jet')

    return _Y
