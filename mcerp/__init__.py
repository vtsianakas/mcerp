"""
================================================================================
mcerp: Real-time latin-hypercube-sampling-based Monte Carlo Error Propagation
================================================================================

Author: Abraham Lee
Copyright: 2013 - 2014
"""
import math
import statsmodels.api as sm
from copy import deepcopy
from pkg_resources import get_distribution, DistributionNotFound
import numpy as np
import multiprocessing as mp
import scipy.stats as ss
from .lhd import lhd

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

__author__ = "Abraham Lee"

npts = 10_000

CONSTANT_TYPES = (float, int, int)


class NotUpcast(Exception):
    """Raised when an object cannot be converted to a number with uncertainty"""


def to_uncertain_func(x):
    """
    Transforms x into an UncertainFunction-compatible object,
    unless it is already an UncertainFunction (in which case x is returned
    unchanged).

    Raises an exception unless 'x' belongs to some specific classes of
    objects that are known not to depend on UncertainFunction objects
    (which then cannot be considered as constants).
    """
    if isinstance(x, UncertainFunction):
        return x

    # ! In Python 2.6+, numbers.Number could be used instead, here:
    elif isinstance(x, CONSTANT_TYPES):
        # No variable => no derivative to define:
        return UncertainFunction([x] * npts)

    raise NotUpcast("%s cannot be converted to a number with" " uncertainty" % type(x))

uncertain=to_uncertain_func

def to_uncertain_list(array):
    """
    Transforms a numpy 2D array into a list of `UncertainFunction` objects.
    """
    assert array.ndim == 2, f"Expected 2D array, got {array.ndim} dimensions."
    l=array.shape[0]
    return [UncertainFunction(array[x]) for x in range(l)]

class UncertainFunction:
    """
    UncertainFunction objects represent the uncertainty of a result of
    calculations with uncertain variables. Nearly all basic mathematical
    operations are supported.

    This class is mostly intended for internal use.
    """

    def __init__(self, mcpts):
        self._mcpts = np.atleast_1d(mcpts).flatten()
        self.tag = None

    def round(digits: int = 2):
        self._mcpts = self._mcpts.round(digits)


    def where(self, statement: str = None) -> np.typing.ArrayLike:
        """
        Single where statement

        Parameters
        ----------
        statement : str
            The where clause

        Examples
        --------

            >>> probA = Bern(.3)
            >>> probA.where(">0")
            array([   3,    7,    9, ..., 9989, 9993, 9996])

        You *cannot* use chained statements such as logical `and`s and others.
        """
        assert statement is not None, "`statement` is required."
        assert isinstance(statement, str), f"`statement` must be a string, got {type(statement)}"
        sel = f"self._mcpts {statement}"
        return np.where(eval(sel))[0]


    def _multiply_process(self, other, ranges):
        drng = np.random.default_rng(ranges[0])
        shape = ranges[1] - ranges[0]
        partial = np.empty(shape)
        if isinstance(other, UncertainFunction):
            for i, idx in zip(range(ranges[0], ranges[1]), range(shape)):
                partial[idx] = np.sum(drng.choice(self._mcpts, int(other[i]), False))
        elif isinstance(other, int):
            for i, idx in zip(range(ranges[0], ranges[1]), range(shape)):
                partial[idx] = np.sum(drng.choice(self._mcpts, other, False))
        else:
            raise ValueError(f"`other` should be int | UncertainFunction. Got {type(other)}")
        return partial

    def multiply(self, other):
        """
        Muplitplies `self` by `other` using parallel processing.

        The computation is divided into chunks, and each chunk is processed in parallel using Python's
        multiprocessing Pool.

        Parameters
        ----------
        other : UncertainFunction or int
            The object to multiply with. If UncertainFunction, element-wise sampling is performed based
            on the values in 'other'. If `int`, a fixed number of samples is taken for each multiplication.

        Returns
        -------
        UncertainFunction
            A new UncertainFunction instance containing the results of the multiplication.

        Raises
        ------
        ValueError
            If 'other' is neither an instance of UncertainFunction nor int.

        Notes
        -----
        To understand the notion of the word `multiply` as used here, consider the following example:

        A ship's hull consists of 562 plates, each of which has to be riveted in place. Each plate is riveted by
        one worker. The supervisor, reflecting on the efficiency of her workforce, considers that the best her
        riveters have ever done is put up a plate in 3 h 45 min. On the other hand, the longest they have taken is
        about 5h 30min, and it is far more likely that a plate would be riveted in about 4 h 15 min. Each
        riveter is paid £7.50 an hour. What is the total labour cost for riveting? One's first thought might be to
        model the total cost as follows:

        ``Cost = 562 * Tri(3.75, 4.25, 5.5) * 7.50``

        What happens if we run a simulation on this formula? In some iterations we will produce values close
        to 3.75 from the triangular distribution. This is saying that all of the plates could have been riveted in
        record time - clearly not a realistic scenario. Similarly, some iterations will generate values close to 5.5
        from the triangular distribution - the scenario that the workforce took as long to put up every plate on
        the ship as it took them to do the trickiest plate in memory.

        The problem lies in the fact that the triangular distribution is modelling the uncertainty of an *individual*
        plate but we are using it as if it were the distribution of the average time for 562 plates.

        Thus, using the method we would write:

        ``Cost = Tri(3.75, 4.25, 5.5).multiply(562) * 7.50``

        562 values are drown at random from `Tri(3.75, 4.25, 5.5)` and summed together.
        This process is repeated `npts` times and the results are returned as an `UncertainFunction`.
        """
        def get_ranges(lenght, step):
            arr = np.arange(0, lenght, 1)
            pairs = [(arr[i], arr[min(i + step, arr[-1])]) for i in range(0, arr[-1], step)]
            pairs[-1] = (pairs[-1][0], lenght)
            return pairs

        chunk_size = npts // mp.cpu_count()
        ranges = get_ranges(npts, chunk_size)
        params = zip( [other]*len(ranges), ranges )

        with mp.Pool() as pool:
            results = pool.starmap(self._multiply_process, params)

        return UncertainFunction(np.concatenate(results))


    def expand(self, times=0):
        """
        Expands an `UncertainFunction` `times` times.

        Parameters
        ----------

        times : int (default=0)
            How many times to expand an `UncertainFunction`. Consider it to reflect a certain number
            of time-periods, years or other. If default, it returns `self`.
        """
        return np.full((times, self._mcpts.shape[0]), self._mcpts)


    def expand_by_rate(self, rate=None, times=0, padding=0, return_array=True):
        """
        Expands an `UncertainFunction` `times` times by a `rate`.

        Consider the following use case: You have

        Rate could be an `UncertainFunction` itself.

        Parameters
        ----------
        rate : float | `UncertainFunction`
            The rate by which the `UncertainFunction` grows. Could be a percentage, i.e. 0.03, implying a 3%
            growth rate, or an `UncertainFunction`. The rate could be in [0, +inf), thus a rate of 1.51
            implies a 151% growth.
        times : int
            How many time units to expand (days, years, etc.)
        pad : int
            Padding to the return `numpy` array
        return_array : bool
            If `True` (default) the method returns a `numpy.array` with shape `(times, npts)`.
            If `False` the method returns a `list` containing `times` `UncertainFunction`s

        Returns
        -------
        out : list | `numpy.array`
            See `return_array` above
        """
        assert rate is not None, "`rate' cannot be None"
        assert times > 0, "`times' should be greater than 0"
        assert 0 <= padding <= times, "`padding' should be in [0, times]"

        times = times + padding
        if isinstance(rate, UncertainFunction):
            if not np.all(rate>=0):
                raise Exception(f"`rate` should be >= 0, got minimum value {rate.min}.")
            expanded = ((1 + rate._mcpts) ** np.arange(times)[:, np.newaxis]) * self

        elif isinstance(rate, float):
            if not rate >= 0:
                raise Exception(f"`rate` should be >= 0, got {rate}.")
            rates = (1 + rate) ** np.arange(times)
            expanded = self._mcpts * rates[:, np.newaxis]

        else:
            raise Exception(f"`rate` should be `float` or `UncertainFunction`, got {type(rate)}")

        expanded=expanded[padding:]
        if return_array:
            return expanded
        else:
            return list(map(UncertainFunction, expanded))


    def __getitem__(self, idx):
    	return np.array(self._mcpts[idx])

    def __setitem__(self, idx, value):
    	self._mcpts[idx] = value

    @property
    def max(self):
        """
        Max value
        """
        return self._mcpts.max()

    @property
    def min(self):
        """
        Max value
        """
        return self._mcpts.min()

    @property
    def mean(self):
        """
        Mean value as a result of an uncertainty calculation
        """
        mn = np.mean(self._mcpts)
        return mn

    @property
    def var(self):
        """
        Variance value as a result of an uncertainty calculation
        """
        mn = self.mean
        vr = np.mean((self._mcpts - mn) ** 2)
        return vr

    @property
    def std(self):
        r"""
        Standard deviation value as a result of an uncertainty calculation,
        defined as::

                    ________
            std = \/variance

        """
        return self.var ** 0.5

    @property
    def skew(self):
        r"""
        Skewness coefficient value as a result of an uncertainty calculation,
        defined as::

              _____     m3
            \/beta1 = ------
                      std**3

        where m3 is the third central moment and std is the standard deviation
        """
        mn = self.mean
        sd = self.std
        sk = 0.0 if abs(sd) <= 1e-8 else np.mean((self._mcpts - mn) ** 3) / sd ** 3
        return sk

    @property
    def kurt(self):
        """
        Kurtosis coefficient value as a result of an uncertainty calculation,
        defined as::

                      m4
            beta2 = ------
                    std**4

        where m4 is the fourth central moment and std is the standard deviation
        """
        mn = self.mean
        sd = self.std
        kt = 0.0 if abs(sd) <= 1e-8 else np.mean((self._mcpts - mn) ** 4) / sd ** 4
        return kt

    @property
    def stats(self):
        """
        The first four standard moments of a distribution: mean, variance, and
        standardized skewness and kurtosis coefficients.
        """
        mn = self.mean
        vr = self.var
        sk = self.skew
        kt = self.kurt
        return [mn, vr, sk, kt]

    @property
    def cv(self):
    	"""
    	Return the coefficient of variation
    	"""
    	return self.std / self.mean

    def percentile(self, val):
        """
        Get the distribution value at a given percentile or set of percentiles.
        This follows the NIST method for calculating percentiles.

        Parameters
        ----------
        val : scalar or array
            Either a single value or an array of values between 0 and 1.

        Returns
        -------
        out : scalar or array
            The actual distribution value that appears at the requested
            percentile value or values

        """
        try:
            # test to see if an input is given as an array
            out = [self.percentile(vi) for vi in val]
        except (ValueError, TypeError):
            if val <= 0:
                out = float(min(self._mcpts))
            elif val >= 1:
                out = float(max(self._mcpts))
            else:
                tmp = np.sort(self._mcpts)
                n = val * (len(tmp) + 1)
                k, d = int(n), n - int(n)
                out = float(tmp[k] + d * (tmp[k + 1] - tmp[k]))
        if isinstance(val, np.ndarray):
            out = np.array(out)
        return out

    def _to_general_representation(self, str_func):
        mn, vr, sk, kt = self.stats
        return (
            "uv({:}, {:}, {:}, {:})".format(
                str_func(mn), str_func(vr), str_func(sk), str_func(kt)
            )
            if any([vr, sk, kt])
            else str_func(mn)
        )

    def __str__(self):
        return self._to_general_representation(str)

    def __repr__(self):
        return str(self)

    def summary(self, name=None, precision=2):
        """Summary statistics

        Parameters
        ----------
        name : ``str``
            The variable's name to be printed
        precision: ``int``
            Decimal precision for the printable output
        """
        def get_quantiles(self):
            s = ""
            for i in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
                s += f' > P{i*100:.<39.2f}:{np.quantile(self._mcpts, i):>15,.{precision}f}\n'
                # s += f" > P{i*100}................... {np.quantile(self._mcpts, i)}\n"
            return s

        mn, vr, sk, kt = self.stats
        if name is not None:
            s = "MCERP Uncertain Value (" + name + "):\n"
        elif self.tag is not None:
            s = "MCERP Uncertain Value (" + self.tag + "):\n"
        else:
            s = "MCERP Uncertain Value:\n"
        s += f' > {"Mean":.<40}:{mn:>15,.{precision}f}\n'
        s += f' > {"Variance":.<40}:{vr:>15,.{precision}f}\n'
        s += f' > {"St.Dev":.<40}:{self.std:>15,.{precision}f}\n'
        s += f' > {"CV":.<40}:{self.cv:>15,.{precision}f}\n'
        s += f' > {"Skewness Coefficient":.<40}:{sk:>15,.{precision}f}\n'
        s += f' > {"Kurtosis Coefficient":.<40}:{kt:>15,.{precision}f}\n'
        s += f' > {"Min":.<40}:{self.min:>15,.{precision}f}\n'
        s += f' > {"Max":.<40}:{self.max:>15,.{precision}f}\n'
        s += f' > {"Range":.<40}:{self.max-self.min:>15,.{precision}f}\n'
        s += f' > {"IQR":.<40}:{np.quantile(self._mcpts, 0.75) - np.quantile(self._mcpts, 0.25):>15,.{precision}f}\n'
        s += f' > {"Std.Error":.<40}:{self.std/np.sqrt(len(self)):>15,.{precision}f}\n'
        s += get_quantiles(self)
        print(s)

    def describe(self, name=None):
        """
        Cleanly show what the four displayed distribution moments are:
            - Mean
            - Variance
            - Standardized Skewness Coefficient
            - Standardized Kurtosis Coefficient

        For a standard Normal distribution, these are [0, 1, 0, 3].

        If the object has an associated tag, this is presented. If the optional
        ``name`` kwarg is utilized, this is presented as with the moments.
        Otherwise, no unique name is presented.

        Example
        =======
        ::

            >>> x = N(0, 1, 'x')
            >>> x.describe()  # print tag since assigned
            MCERP Uncertain Value (x):
            ...

            >>> x.describe('foobar')  # 'name' kwarg takes precedence
            MCERP Uncertain Value (foobar):
            ...

            >>> y = x**2
            >>> y.describe('y')  # print name since assigned
            MCERP Uncertain Value (y):
            ...

            >>> y.describe()  # print nothing since no tag
            MCERP Uncertain Value:
            ...

         """
        mn, vr, sk, kt = self.stats
        if name is not None:
            s = "MCERP Uncertain Value (" + name + "):\n"
        elif self.tag is not None:
            s = "MCERP Uncertain Value (" + self.tag + "):\n"
        else:
            s = "MCERP Uncertain Value:\n"
        s += " > Mean................... {: }\n".format(mn)
        s += " > Variance............... {: }\n".format(vr)
        s += " > Skewness Coefficient... {: }\n".format(sk)
        s += " > Kurtosis Coefficient... {: }\n".format(kt)
        print(s)

    def get_values_for_spider(
        self,
        variable,
        percentiles=np.array([.01, .10, .25, .50, .75, .90, .99]),
        metric=np.mean,
        report=False,
    ):
        """Returns values to plot a spider graph

        Params
        ------
        variable: `UncertainedFunction`
            The input variable to do the sensitivity analysis against the output variable `self`
        percentiles: `np.array`
            The list of percentiles to be calculated: For example `[.25, .50, .75]`.
        metric: `np.mean` | `np.std` | `np.var` | `np.median`
            The metric to be used. Default is the mean value of the output variable `self`

        Returns
        -------
        out : list
            The list of values of the output variable according to the `metric`
        """
        if metric not in [np.mean, np.std, np.var, np.median]:
            raise ValueError(f"`metric' should be one of `np.mean` | `np.std` | `np.var` | `np.median`. Got {type(metric)}, {metric}")
        variable=UncertainFunction(variable)
        if 0 not in percentiles:
            percentiles = np.insert(percentiles, 0, 0)
        if 1 not in percentiles:
            percentiles = np.append(percentiles, 1)
        p=np.percentile(variable._mcpts, percentiles*100)

        npvs={
            'npv': [],
            'values': [],
            'percentile': [],
            'percentile_value': []
        }
        for i, (start, end) in enumerate(zip(p[:-1], p[1:])):
            if start == 0.0:
                # first percentile, include min value in calculations, range is [start, end]
                values_in_range=np.where((variable._mcpts >= start) & (variable._mcpts <= end))
            else:
                # not the first percentile, range is (start, end]
                values_in_range=np.where((variable._mcpts > start) & (variable._mcpts <= end))
            npvs['npv'].append(metric(self._mcpts[values_in_range]))
            npvs['values'].append(values_in_range[0].size)
            npvs['percentile'].append((percentiles[i],))
            npvs['percentile_value'].append((start, end))

        return npvs if report else npvs['npv']


    def get_negative_percentile(self, variable=None, percentiles=np.arange(0,101,1)/100):
        """
        Calculates the percentile at which the Net Present Value (NPV) becomes negative for a given input variable.

        This method identifies the first occurrence of a negative NPV value by iterating over the provided
        percentiles and corresponding NPV values. It returns the NPV, the number of simulated values for
        the input variable percentile, and the corresponding critical value bellow which the NPV becomes,
        on average, negative.

        Parameters
        ----------
        variable : `UncertainFunction`, required
            The input variable for which the negative NPV percentile is to be calculated.

        percentiles : array-like, optional
            The array of percentiles at which to calculate the NPV values. By default, it calculates for
            percentiles from 0% to 100% with an interval of 1% (i.e., `np.arange(0, 101, 1) / 100`).

        Returns
        -------
        dict
            A dictionary containing the following keys:
            - 'npv': The NPV value at the first occurrence of a negative NPV, or `None` if no negative NPV is found.
            - 'values': The number of simulated values of the input variable with which the calcluations are made,
            or `None` if no negative NPV is found.
            - 'percentile': The percentile at which the NPV is negative, or `None` if no negative NPV is found.
            - 'critical': The percentile value bellow which the NPV becomes, on average, negative, or `None` if no
            negative NPV is found.

        Notes
        -----
        - The method assumes that `self.get_values_for_spider()` is available to fetch the necessary data for
          the specified variable and percentiles.
        - If no negative NPV is found (i.e., all NPVs are positive), the method returns `None` for all
        dictionary values.
        - The method works by iterating over the percentiles and checking the corresponding NPV values
        until a negative value is found.
        """

        d=self.get_values_for_spider(variable, percentiles=percentiles, report=True)
        idx = None

        if any(x<0 for x in d['npv']) and any(x>0 for x in d['npv']):
            idx = np.argmin(np.abs(d['npv']))
        else:
            return {
                'npv': None,
                'values': None,
                'percentile': None,
                'critical': None
            }

        return {
            'npv': d['npv'][idx],
            'values': d['values'][idx],
            'percentile': d['percentile'][idx],
            'critical': d['percentile_value'][idx][1]
        }

    def plot_percentiles_effect(
        self,
        variable=None,
        var_name=None,
        start=None,
        stop=None,
        title=None,
        a=0.01,
        **kwargs
    ):
        """
        Plots the effect of values between two specified percentiles of an input variable on the simulation's output.

        This function calculates the specified percentiles of the given input variable and then plots
        the probability density function (PDF) of the simulation's output, given the values of the input variable.
        The plot is customized with options provided in `kwargs`.

        Parameters:
        ----------
        variable : `UncertainFunction`, required
            The input variable to be analyzed.

        var_name : str, optional
            The name of the `variable` to be displayed in the title of the plot. If not provided, defaults to
            an empty string.

        start : int, required
            The lower bound percentile (inclusive) for the range of values to analyze.

        stop : int, required
            The upper bound percentile (exclusive) for the range of values to analyze.
            Must be greater than `start`.

        title : str, optional
            A custom title for the plot. If not provided, a default title is generated based on `var_name`,
            `start`, and `stop`.

        a : float, optional
            The significance level for the plot. Defaults to `0.01`.

        precision : int, optional
            The number of decimal places to display for percentile values in the title. Defaults to `2`.

        kwargs : keyword arguments
            Additional keyword arguments that are passed to the plotting function (`plot_pdf` of `UncertainFunction`).
            Can include:
            - 'label' : str, default 'NPV'
                The label for the plot's legend.
            - 'xlabel' : str, default 'EUR'
                The label for the x-axis.
            - 'x_label_formatter' : function, default `x`
                A function to format the x-axis tick labels.

        Raises:
        ------
        ValueError :
            If `start` and `stop` are not integers or if `start` is not smaller than `stop`.

        Notes:
        -----
        - The function uses `np.percentile` (see the official NumPy documentation here:
          https://numpy.org/doc/stable/reference/generated/numpy.percentile.html) to calculate the values
          corresponding to the specified percentiles for the variable's Monte Carlo points (`_mcpts`).
        - The function generates a plot of the PDF of the values between the `start` and `stop` percentiles.
        - The `x_label_formatter` function is used to format the x-axis tick labels. For more information on custom
          tick label formatting in Matplotlib, see the official documentation for `FuncFormatter` here:
          https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.FuncFormatter
        """

        if not isinstance(start, int) or not isinstance(stop, int):
            raise ValueError("Both 'start' and 'stop' must be integers.")

        if start >= stop:
            raise ValueError("'start' must be smaller than 'stop'.")

        kwargs['label'] = kwargs.get('label', 'NPV')
        kwargs['xlabel'] = kwargs.get('xlabel', 'EUR')

        precision = kwargs['precision'] = kwargs.get('precision', 2)
        def x(x, pos):
            return f"{x:,.0f}"
        kwargs['x_label_formatter'] = kwargs.get('x_label_formatter', x)

        percentile_low = np.percentile(variable._mcpts, start)
        percentile_high = np.percentile(variable._mcpts, stop)

        if var_name is None:
            var_name = ""
        idx=np.where((variable._mcpts > percentile_low)&(variable._mcpts <= percentile_high))
        if title is None:
            title = f"""{var_name} Effect on NPV
            Bewteen Percentiles {start} and {stop}
            Variable values in $({percentile_low:,.{precision}f}, {percentile_high:,.{precision}f}]$"""
        else:
            title += f"""
            Bewteen Percentiles {start} and {stop}
            Variable values in $({percentile_low:,.{precision}f}, {percentile_high:,.{precision}f}]$"""
        UncertainFunction(self._mcpts[idx]).plot_pdf(name=title, a=a, **kwargs)


    def plot_approximate_break_even(
        self,
        variable=None,
        var_name=None,
        title=None,
        a=0.01,
        precision=2,
        **kwargs
    ):
        """
        Plots the effect of an input variable on the simulation's output, using values from the first percentile that
        the simulation's output mean is negative.

        This function calculates the first negative percentile of the given variable, which represents the first
        percentile at which the NPV is negative. It then calls `plot_percentiles_effect` to generate a plot
        showing the effect of the variable on NPV for the calculated negative percentile range.

        Parameters:
        ----------
        self : object, optional
            The object instance containing the Monte Carlo simulation points, defaulting to `npv`.

        variable : object, required
            The variable containing the Monte Carlo simulation points to be analyzed. It must be compatible with the
            `plot_percentiles_effect` and `plot_percentiles_for_variable` functions.

        var_name : str, optional
            The name of the variable to be displayed in the title of the plot. If not provided, defaults to an empty string.

        start : int, optional
            The starting percentile (inclusive) for plotting the variable's effect on NPV. This value is automatically
            calculated as the first negative percentile from `plot_percentiles_effect`.

        stop : int, optional
            The ending percentile (exclusive) for plotting the variable's effect on NPV. This value is automatically
            calculated as one percentile higher than `start`.

        title : str, optional
            A custom title for the plot. If not provided, a default title is generated based on `var_name`, `start`,
            and `stop`.

        a : float, optional
            The significance level for the plot. Defaults to 0.01.

        precision : int, optional
            The number of decimal places to display for percentile values in the title. Defaults to 2.

        kwargs : keyword arguments
            Additional keyword arguments passed to `plot_percentiles_for_variable` for customizing the plot.
            These could include:
            - 'xlabel' : str, optional
                The label for the x-axis. Defaults to 'EUR'.
            - 'label' : str, optional
                The label for the plot's legend. Defaults to 'NPV'.
            - 'x_label_formatter' : function, optional
                A custom function to format the x-axis tick labels. Defaults to `None`.

        Returns:
        -------
        None

        Notes:
        -----
        - This function internally calls `plot_percentiles_effect` to calculate the first negative percentile, and then
          generates a plot showing the variable's effect on NPV between the `start` and `stop` percentiles.
        - If no negative percentile can be found, the function may not behave as expected, as it depends on the output
          of `plot_percentiles_effect`.

        Example:
        --------
        npv.plot_negative_percentile(variable=revenues)
        # This will generate a plot showing the effect of the variable on NPV at the first negative percentile.
        """
        if self.get_negative_percentile(variable)['percentile'] is not None:
            start = int(self.get_negative_percentile(variable)['percentile'][0]*100)
        else:
            raise ValueError(f"No percentile of the input `variable` gives an average zero of the output variable")
        stop = start+1
        kwargs['precision'] = precision
        self.plot_percentiles_effect(
            variable=variable,
            var_name=var_name,
            start=start,
            stop=stop,
            a=a,
            title=title,
            **kwargs
        )



    def rank_correlation(self, variables=None):
        """
        Returns the rank correlation of ``variables`` with respect to ``target``

        Params
        ------
        variables: array-like of type ``UncertainValue``
            The variables to be rank correlated
        target: ``UncertainValue``
            The target variable

        Returns
        -------

        out: ``numpy`` array
            The correllation vector (1D)
        """
        c=np.array([])
        for i in variables:
            c=np.append(c, stats.spearmanr(i, self).statistic)
        return c.round(2)

    def contribution_to_variance(self, variables=None):
        """
        Returns the "Contribution of ``variables`` to the Variance" of ``self``.

        Calculates the contribution to the variance by squaring the rank correlation coefficients and normalizing
        them to 100%. The same method is used  by CrystallBall(R) by ORACLE. The method fails miserably when the
        variables are correlated or when the model is non-linear.

        Params
        ------
        variables: array-like with elements of type ``UncertainValue``
            The variables to be rank correlated

        Returns
        -------

        out: ``numpy`` array
            The contribution to variance as percentage, normalized to 100%
        """
        cors=self.rank_correlation(variables=variables)
        cors2=cors**2
        norm=((cors2/cors2.sum())*100).round(2)
        norm[np.where(cors<0)] = norm[np.where(cors<0)]*-1
        return norm

    def stepwise_regression(self, variables, limit=5_000):
        if limit > npts:
            limit = npts

        for key, value in variables.items():
            variables[key] = value[:limit]

        vars_keys = set(variables.keys())
        selected = []

        # Forward stepwise regression
        while True:
            max_key=None
            rsquared=0
            if len(vars_keys) == 0:
                break
            keys = vars_keys - set(selected)

            for key in keys:
                sel = [variables[x] for x in selected]
                sel.append(variables[key])
                X=np.vstack(sel)
                y=self._mcpts[:limit]

                rsq = sm.OLS(y, X.T).fit().rsquared

                if rsq > rsquared:
                    max_key = key
                    rsquared = rsq

            if max_key is not None:
                selected.append(max_key)
                vars_keys = vars_keys - {max_key}
            else:
                vars_keys = vars_keys - keys

        output={}

        # Forward regression
        for i in range(len(selected)):
            sel = [variables[x] for x in selected[0:i+1]]
            X=np.vstack(sel)

            result=sm.OLS(self._mcpts[:limit], X.T).fit()
            rsq = result.rsquared
            output[selected[i]]  = rsq

        out = np.array(list(output.values()))
        first = out[0]
        out = np.concatenate([[first], np.diff(out)])
        changes=dict()
        for key, value in zip(output, list(out)):
            changes[key] = np.round(value*100, 2)

        return changes


    def plot_variance(self, variables, ax=None, fig=None, method='rank', **kwargs):
        """
        Plots a "Contribution to Variance" tornado graph.

        Plot the "Contribution of ``variables`` to the Variance" of ``self``.

        The contribution to variance is calculated by squaring the rank correlation coefficients and normalizing
        them to 100%. The same method is used  by CrystallBall(R) by ORACLE. The method fails miserably when the
        variables are correlated or when the model is non-linear.

        Params
        ------

        variables: ``dict``
            A ``dict`` with keys ``str`` to display in the graph and values their corresponding ``UncertainedVariable``s
        ax: ``matplotlib.pyplot.axis``
            To be printed on an axis
        method: ``rank``|``reg``
            If type is ``rank`` (default) the rank correlation method is used. If type is ``reg`` the regression method
            is used.
        kwargs: ``dict``
            Any valid ``matplotlib.pyplot.bar`` ``kwargs``
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        assert method in ['rank', 'reg'], f"`method' should be in ['rank', 'reg']. Got: {method}"
        params = {
            'text.usetex' : True,
            'font.serif' : 'cmr10, Computer Modern Serif, DejaVu Serif',
            'font.family' : 'serif',
            'axes.formatter.use_mathtext' : True,
            'mathtext.fontset' : 'cm',
        }
        mpl.rcParams.update(params)
        if ax is None:
            if kwargs.get('figsize', None) is not None:
                fig, ax = plt.subplots(figsize=kwargs.get('figsize',(10, 8)))
                del kwargs['figsize']
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
        if method == 'rank':
            method_str="Rank Correlation"
            ctv = self.contribution_to_variance(variables.values())
            ctv = dict(zip(variables.keys(), ctv))
            ctv = dict(sorted(ctv.items(), key=lambda item: item[1], reverse=False))
            max_x = max(ctv.values()) + 12.5
            min_x = min(ctv.values()) - 12.5
            ax.set_xlim(min_x, max_x)

        elif method == 'reg':
            raise ValueError(f"Not Implemented. Consider not assuming linear relationships.")
            method_str="Regression"
            ctv = self.stepwise_regression(variables)
            ctv = dict(sorted(ctv.items(), key=lambda item: item[1], reverse=False))
            ax.set_xlim(right=max(ctv.values()) + 12.5)


        params=dict(color='#182234', align='center', alpha=0.3)
        for key in params:
            if key in kwargs:
                params[key] = kwargs[key]
                del kwargs[key]
        bars=[]
        for param, value in ctv.items():
            b=ax.barh(param, value, **params, **kwargs)
            bars.append(b)

        for bar, value in zip(bars, ctv.values()):
            ax.bar_label(bar, labels=[fr"{value:.2f}\%"], label_type='edge', padding=8, weight='bold')

        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(False)
        ax.axvline(
            0,
            linestyle=':',
            color='#182234',
        )
        fig.text(0.5, 1.02, "Contribution to Variance", fontsize=24, transform=ax.transAxes, ha='center', va='bottom')
        fig.text(0, 1.001, f"Method: {method_str}", transform=ax.transAxes, ha='left', va='bottom')

        ax.set_xlabel("Percentage points", fontsize=15)
        ax.set_ylabel("Variable", fontsize=15)
        fig.tight_layout()

    def plot_spider(
         self,
         variables=None,
         name='NPV',
         ax=None,
         fig=None,
         figsize=(14, 10),
         fontsize=16,
         grid=True,
         xlabel='Input Distribution Percentiles',
         ylabel="NPV",
         legend=True,
         percentiles=np.array([.01, .05, .25, .50, .75, .95, .99,]),
         metric=np.mean,
         y_label_formatter=None,
         **kwargs
    ):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedLocator, MaxNLocator

        if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=figsize)
        elif ax is None or fig is None:
            raise ValueError(f"You should provide both an `ax` and `fig` arguments")

        ax.xaxis.set_major_locator(FixedLocator(np.arange(0,105,5)))
        ax.xaxis.set_minor_locator(FixedLocator(np.arange(0,101,1)))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))

        def y_major_formatter(x, pos):
            return fr'${x:,.0f}$'
        if y_label_formatter is None:
            ax.yaxis.set_major_formatter(y_major_formatter)
        else:
            ax.yaxis.set_major_formatter(y_label_formatter)
        percentiles = np.sort(percentiles)
        if 0 not in percentiles:
            percentiles = np.insert(percentiles, 0, 0)
        if 1 not in percentiles:
            percentiles = np.append(percentiles, 1)

        for key, value in variables.items():
            ax.plot(percentiles[1:]*100, self.get_values_for_spider(value, percentiles, metric=metric), label=key, lw=1.1)
        if legend:
            plt.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0, fontsize=fontsize-2)

        metric_dict = {
            np.mean: 'Mean',
            np.std: 'Standard Deviation',
            np.var: 'Variance',
            np.median: 'Median'
        }

        if metric not in metric_dict.keys():
            raise ValueError(f"`metric` should be one of np.mean | np.std | np.var | np.median. Got {metric}")
        metric_str = metric_dict[metric]

        ax.axhline(metric(self._mcpts), color="black", ls=":", lw=2, label=f"{name} {metric_str}")
        fig.suptitle(f"{name}", fontsize=fontsize+6)
        ax.set_title(f"Change in Output {metric_str} Accross Input Distribution Percentiles", fontsize=fontsize+4)
        ax.set_ylabel(ylabel, fontsize=fontsize+2)
        ax.set_xlabel(xlabel, fontsize=fontsize+2)
        ax.grid(grid)
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.tight_layout()



    def get_positive_percentile(
        self,
        variable=None,
        a=0.01,
        percentiles=np.arange(0,101,1)/100,
        return_int=True,
        **kwargs
    ):
        """
        Finds the first percentile range of the input variable where the `self`'s values are positive with confidence
        `1-a`.

        This method iterates over consecutive percentiles and checks whether the values
        in the specified range are positive, based on the provided confidence level `a`.
        It returns the first pair of percentiles where average `self` values is positive.

        Parameters:
        ----------

        variable : `UncertainFunction`, required
            The input variable to be analyzed.

        a : float, optional
            The significance level used to calculate the threshold for determining positivity.
            Defaults to `0.01`.

        percentiles : array-like, optional
            An array of percentiles to be used for calculating the percentile ranges.
            Defaults to `np.arange(0, 101, 1) / 100`.

        return_int : bool, optional
            If `True`, the function will return the percentiles as integers (in percentage format).
            If `False`, it will return the percentiles as floating-point values.
            Defaults to `True`.

        kwargs : keyword arguments, optional
            Additional keyword arguments that may be passed to the function but are not used in
            the current implementation.

        Returns:
        -------
        tuple
            If `return_int` is `True`, returns a tuple of two integers representing the lower
            and upper percentiles where `self`'s values are positive (in percentage format).
            If `return_int` is `False`, returns the original floating-point pair of percentiles.

        Raises:
        ------
        ValueError
            If the number of percentiles is not an odd number (i.e., `percentiles.shape[0] % 2 != 1`).

        Notes:
        -----
        - The function splits the percentiles into consecutive pairs and calculates the low and high percentiles
          for each pair in the variable's data (`_mcpts`).
        - The positivity condition is determined by checking if the percentile corresponding to `a/2` is greater
        than or equal to zero.
        - The function will return the first pair of percentiles where the values are positive based on the threshold.

        Example:
        --------
        result = npv.get_positive_percentile(variable=my_variable, a=0.05)
        # This will return the first pair of percentiles where `npv` values exceed the positivity threshold.
        """

        if percentiles.shape[0] % 2 != 1:
            raise ValueError("`percentiles` are divisible by 2")
        pairs = list(zip(percentiles[:-1], percentiles[1:]))

        for i in pairs:
            percentile_low = np.percentile(variable._mcpts, i[0]*100)
            percentile_high = np.percentile(variable._mcpts, i[1]*100)

            idx = np.where(
                (variable._mcpts > percentile_low) & (variable._mcpts <= percentile_high)
            )
            output = self[idx]
            if np.percentile(output, (a/2)*100) >= 0:
                if return_int:
                    return int(i[0]*100), int(i[1]*100),
                else:
                    return i


    def plot_pdf(
        self,
        name=None,
        a=0.05,
        ax=None,
        fig=None,
        figsize=(10, 5),
        fontsize=16,
        bins='sturges',
        grid=True,
        xlabel='Values',
        ylabel="Probability Density",
        samples=50,
        hist=True,
        legend=True,
        color_cycler=None,
        x_label_formatter=None,
        y_label_formatter=None,
        x_major_locator=None,
        **kwargs
    ):
        import matplotlib.pyplot as plt
        from cycler import cycler
        from matplotlib.ticker import FuncFormatter, FixedLocator

        precision = kwargs.get('precision', 2)

        if color_cycler is not None:
            if not isinstance(color_cycler, list):
                raise ValueError(f"`color_cycler` should be a list of colors")
            elif len(color_cycler) == 0:
                raise ValueError(f"`color_cycler` should not be an empty list")

        if color_cycler is None:
            cy = cycler(
                'color',
                ['#182234', '#CC0E80', '#70BF4B', '#F2B035', '#F20505']
            )
        else:
            cy = cycler(
                'color',
                color_cycler
            )

        if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=figsize)
        elif ax is None or fig is None:
            raise ValueError(f"You should provide both an `ax` and `fig` arguments")

        ax.set_prop_cycle(cy)

        # Get the first color in case a histogram will not be drawn
        initial_color=next(iter(cy))

        hist_line = None
        if hist:
            ax.hist(
                self._mcpts,
                bins=bins,
                density=True,
                alpha=kwargs.get('alpha', 0.3),
                edgecolor='black',
                align='mid',
            )

        # If hist is not drawn, then draw the kde line using the cycler's first color.
        # If hist is drawn, then draw the kde line using the same color.
        color=initial_color['color']

        p = ss.gaussian_kde(self._mcpts)
        xp = np.linspace(self.min, self.max, samples)
        y = p.evaluate(xp)
        # Plot the empirical pdf line
        ax.plot(xp, y, ls='-', label=kwargs.get('label',"Empirical PDF"), color=color, lw=2, alpha=1)

        # Plot the mean
        ax.axvline(
            self.mean,
            color=color,
            linestyle='dashed',
            alpha=1.0,
            lw=1.3,
            label=rf"$\bar{{\mu}} = {self.mean:,.{precision}f}$"
        )

        # Plot confidence lines
        ci_low, ci_high = np.percentile(self._mcpts, np.array([a/2, 1-a/2])*100)
        alpha_low = f"{f'{(a/2)*100:,.4f}'.rstrip('0')}"
        alpha_high = f"{f'{(1-a/2)*100:,.4f}'.rstrip('0')}"
        ax.axvline(
            ci_low,
            linestyle=':',
            color=color,
            label=rf"$P_{{{alpha_low}}} = {ci_low:,.{precision}f}$",
            alpha=kwargs.get('alpha', 0.8)
        )
        ax.axvline(
            ci_high,
            linestyle=':',
            color=color,
            label=rf"$P_{{{alpha_high}}} = {ci_high:,.{precision}f}$",
            alpha=kwargs.get('alpha', 0.8)
        )
        ax.axvspan(ci_low, ci_high, color=color, alpha=0.05)

        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize+1)
        ax.set_ylabel(ylabel, fontsize=fontsize+1)

        if x_label_formatter:
            ax.xaxis.set_major_formatter(FuncFormatter(x_label_formatter))
        if y_label_formatter:
            ax.yaxis.set_major_formatter(FuncFormatter(y_label_formatter))
        if x_major_locator == 'tri':
            ax.xaxis.set_major_locator(FixedLocator([
                ci_low, self.mean, ci_high
            ]))
        elif x_major_locator:
            ax.xaxis.set_major_locator(x_major_locator)

        if name is not None:
            ax.set_title(f"{name}", fontsize=fontsize+4)
        elif self.tag is not None:
            ax.set_title(f"{self.initial_colortag}", fontsize=fontsize+4)
        else:
            ax.set_title("Histogram", fontsize=fontsize+4)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend(fontsize=fontsize-2)

    def plot_cdf(
        self,
        name=None,
        a=0.05,
        ax=None,
        fig=None,
        figsize=(10, 5),
        fontsize=16,
        grid=True,
        xlabel='Values',
        ylabel='Cum. Probability',
        color_cycler=None,
        x_label_formatter=None,
        y_label_formatter=None,
        x_major_locator=None,
        **kwargs
    ):
        import matplotlib.pyplot as plt
        from cycler import cycler
        from matplotlib.ticker import FuncFormatter

        if color_cycler is not None:
            if not isinstance(color_cycler, list):
                raise ValueError(f"`color_cycler` should be a list of colors")
            elif len(color_cycler) == 0:
                raise ValueError(f"`color_cycler` should not be an empty list")

        precision = kwargs.pop('precision', 2)

        if color_cycler is None:
            cy = cycler(
                'color',
                ['#182234', '#CC0E80', '#70BF4B', '#F2B035', '#F20505']
            )
        else:
            cy = cycler(
                    'color',
                    color_cycler
            )

        if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=figsize)
        elif ax is None or fig is None:
            raise ValueError(f"You should provide both an `ax` and `fig` arguments")

        ax.set_prop_cycle(cy)
        data_sorted = np.sort(self._mcpts)

        # calculate the proportional values of samples
        p = 1. * np.arange(len(data_sorted)) / (len(data_sorted) - 1)

        line=ax.plot(data_sorted, p, lw=2, **kwargs)
        ax.set_xlabel(xlabel, fontsize=fontsize+1)
        ax.set_ylabel(ylabel, fontsize=fontsize+1)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        color=line[0].get_color()

        ci_low, ci_high = np.percentile(self._mcpts, np.array([a/2, 1-a/2])*100)
        # ci_low, ci_high = self.percentile([a/2, 1-a/2])
        alpha_low = f"{f'{(a/2)*100:,.4f}'.rstrip('0')}"
        alpha_high = f"{f'{(1-a/2)*100:,.4f}'.rstrip('0')}"
        ax.axvline(
            ci_low,
            linestyle=':',
            color=color,
            label=rf"$P_{{{alpha_low}}} = {ci_low:,.{precision}f}$"
        )
        ax.axvline(
            ci_high,
            linestyle=':',
            color=color,
            label=rf"$P_{{{alpha_high}}} = {ci_high:,.{precision}f}$"
        )
        ax.axvspan(ci_low, ci_high, color=color, alpha=kwargs.get('alpha', 0.05))

        ax.legend(fontsize=fontsize-2)

        title = f""
        if self.tag and name is None:
            title = f"{self.tag}"
        elif name is not None:
            title = f"{name}"
        else:
            title = f"CDF"
        ax.grid(grid)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.set_title(title, fontsize=fontsize+4)
        if x_label_formatter:
            ax.xaxis.set_major_formatter(FuncFormatter(x_label_formatter))
        if y_label_formatter:
            ax.yaxis.set_major_formatter(FuncFormatter(y_label_formatter))
        if x_major_locator == 'tri':
            ax.xaxis.set_major_locator(FixedLocator([
                ci_low, self.mean, ci_high
            ]))
        elif x_major_locator:
            ax.xaxis.set_major_locator(x_major_locator)


    def plot(self, hist=False, show=False, **kwargs):
        """
        Plot the distribution of the UncertainFunction. By default, the
        distribution is shown with a kernel density estimate (kde).

        Optional
        --------
        hist : bool
            If true, a density histogram is displayed (histtype='stepfilled')
        show : bool
            If ``True``, the figure will be displayed after plotting the
            distribution. If ``False``, an explicit call to ``plt.show()`` is
            required to display the figure.
        kwargs : any valid matplotlib.pyplot.plot or .hist kwarg

        """
        import matplotlib.pyplot as plt

        vals = self._mcpts
        low = min(vals)
        high = max(vals)

        p = ss.kde.gaussian_kde(vals)
        xp = np.linspace(low, high, 100)

        if hist:
            h = plt.hist(
                vals,
                bins=int(np.sqrt(len(vals)) + 0.5),
                histtype="stepfilled",
                #normed=True, deprecated
                density=True,
                **kwargs
            )
            plt.ylim(0, 1.1 * h[0].max())
        else:
            plt.plot(xp, p.evaluate(xp), **kwargs)

        plt.xlim(low - (high - low) * 0.1, high + (high - low) * 0.1)

        if show:
            self.show()

    def show(self):
        import matplotlib.pyplot as plt

        plt.show()


    def __add__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[0]._mcpts + uf[1]._mcpts
        return UncertainFunction(mcpts)

    def __radd__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[0]._mcpts + uf[1]._mcpts
        return UncertainFunction(mcpts)

    def __mul__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[0]._mcpts * uf[1]._mcpts
        return UncertainFunction(mcpts)

    def __rmul__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[0]._mcpts * uf[1]._mcpts
        return UncertainFunction(mcpts)

    def __sub__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[0]._mcpts - uf[1]._mcpts
        return UncertainFunction(mcpts)

    def __rsub__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[1]._mcpts - uf[0]._mcpts
        return UncertainFunction(mcpts)

    def __div__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[0]._mcpts / uf[1]._mcpts
        return UncertainFunction(mcpts)

    def __rdiv__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[1]._mcpts / uf[0]._mcpts
        return UncertainFunction(mcpts)

    def __truediv__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[0]._mcpts / uf[1]._mcpts
        return UncertainFunction(mcpts)

    def __rtruediv__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[1]._mcpts / uf[0]._mcpts
        return UncertainFunction(mcpts)

    def __pow__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[0]._mcpts ** uf[1]._mcpts
        return UncertainFunction(mcpts)

    def __rpow__(self, val):
        uf = list(map(to_uncertain_func, [self, val]))
        mcpts = uf[1]._mcpts ** uf[0]._mcpts
        return UncertainFunction(mcpts)

    def __neg__(self):
        mcpts = -self._mcpts
        return UncertainFunction(mcpts)

    def __pos__(self):
        mcpts = self._mcpts
        return UncertainFunction(mcpts)

    def __abs__(self):
        mcpts = np.abs(self._mcpts)
        return UncertainFunction(mcpts)

    def __eq__(self, val):
        """
        If we are comparing two distributions, check the resulting moments. If
        they are the same distribution, then the moments will all be zero and
        we can know that it is actually the same distribution we are comparing
        ``self`` to, otherwise, at least one statistical moment will be non-
        zero.

        If we are comparing ``self`` to a scalar, just do a normal comparison
        so that if the underlying distribution looks like a PMF, a meaningful
        probability of self==val is returned. This can still work quite safely
        for PDF distributions since the likelihood of comparing self to an
        actual sampled value is negligible when mcerp.npts is large.

        Examples:

            >>> h = H(50, 5, 10)  # Hypergeometric distribution (PMF)
            >>> h==4  # what's the probability of getting 4 of the 5?
            0.004
            >>> sum([h==i for i in (0, 1, 2, 3, 4, 5)])  # sum of all discrete probabilities
            1.0

            >>> n = N(0, 1)  # Normal distribution (PDF)
            >>> n==0  # what's the probability of being exactly 0.0?
            0.0
            >>> n>0  # greater than 0.0?
            0.5
            >>> n<0  # less than 0.0?
            0.5
            >>> n==1  # exactly 1.0?
            0.0
        """
        if isinstance(val, UncertainFunction):
            diff = self - val
            return not (diff.mean or diff.var or diff.skew or diff.kurt)
        else:
            return len(self._mcpts[self._mcpts == val]) / float(npts)

    def __ne__(self, val):
        if isinstance(val, UncertainFunction):
            return not self == val
        else:
            return 1 - (self == val)

    def __lt__(self, val):
        """
        If we are comparing two distributions, perform statistical tests,
        otherwise, calculate the probability that the distribution is
        less than val
        """
        if isinstance(val, UncertainFunction):
            tstat, pval = ss.ttest_ind(self._mcpts, val._mcpts, equal_var=False, alternative='less')
            sgn = np.sign(tstat)
            if pval > 0.05:  # Since, statistically, we can't really tell
                return False
            else:
                return True if sgn == -1 else False
        else:
            return self._mcpts[self._mcpts < val].shape[0] / float(self._mcpts.shape[0])

    def __le__(self, val):
        if isinstance(val, UncertainFunction):
            return self < val  # since it doesn't matter to the test
        else:
            return self._mcpts[self._mcpts <= val].shape[0] / float(self._mcpts.shape[0])

    def __gt__(self, val):
        """
        If we are comparing two distributions, perform statistical tests,
        otherwise, calculate the probability that the distribution is
        greater than val
        """
        if isinstance(val, UncertainFunction):
            tstat, pval = ss.ttest_ind(self._mcpts, val._mcpts, equal_var=False, alternative='greater')
            sgn = np.sign(tstat)
            if pval > 0.05:  # Since, statistically, we can't really tell
                return False
            else:
                return True if sgn == 1 else False
        else:
            return 1 - (self <= val)

    def __ge__(self, val):
        if isinstance(val, UncertainFunction):
            return self > val
        else:
            return 1 - (self < val)

    def __bool__(self):
        return not (1 - ((self > 0) + (self < 0)))

    def __len__(self):
        """Return the ``len()``

        There are numerous cases where the ``len()`` function must return a value.
        """
        return self._mcpts.shape[0]

class UncertainVariable(UncertainFunction):
    """
    UncertainVariable objects track the effects of uncertainty, characterized
    in terms of the first four standard moments of statistical distributions
    (mean, variance, skewness and kurtosis coefficients). Monte Carlo simulation,
    in conjunction with Latin-hypercube based sampling performs the calculations.

    Parameters
    ----------
    rv : scipy.stats.distribution
        A distribution to characterize the uncertainty

    tag : str, optional
        A string identifier when information about this variable is printed to
        the screen

    Notes
    -----

    The ``scipy.stats`` module contains many distributions which we can use to
    perform any necessary uncertainty calculation. It is important to follow
    the initialization syntax for creating any kind of distribution object:

        - *Location* and *Scale* values must use the kwargs ``loc`` and
          ``scale``
        - *Shape* values are passed in as non-keyword arguments before the
          location and scale, (see below for syntax examples)..

    The mathematical operations that can be performed on uncertain objects will
    work for any distribution supplied, but may be misleading if the supplied
    moments or distribution is not accurately defined. Here are some guidelines
    for creating UncertainVariable objects using some of the most common
    statistical distributions:

    +---------------------------+-------------+-------------------+-----+---------+
    | Distribution              | scipy.stats |  args             | loc | scale   |
    |                           | class name  | (shape params)    |     |         |
    +===========================+=============+===================+=====+=========+
    | Normal(mu, sigma)         | norm        |                   | mu  | sigma   |
    +---------------------------+-------------+-------------------+-----+---------+
    | Uniform(a, b)             | uniform     |                   | a   | b-a     |
    +---------------------------+-------------+-------------------+-----+---------+
    | Exponential(lamda)        | expon       |                   |     | 1/lamda |
    +---------------------------+-------------+-------------------+-----+---------+
    | Gamma(k, theta)           | gamma       | k                 |     | theta   |
    +---------------------------+-------------+-------------------+-----+---------+
    | Beta(alpha, beta, [a, b]) | beta        | alpha, beta       | a   | b-a     |
    +---------------------------+-------------+-------------------+-----+---------+
    | Log-Normal(mu, sigma)     | lognorm     | sigma             | mu  |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Chi-Square(k)             | chi2        | k                 |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | F(d1, d2)                 | f           | d1, d2            |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Triangular(a, b, c)       | triang      | c                 | a   | b-a     |
    +---------------------------+-------------+-------------------+-----+---------+
    | Student-T(v)              | t           | v                 |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Weibull(lamda, k)         | exponweib   | lamda, k          |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Bernoulli(p)              | bernoulli   | p                 |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Binomial(n, p)            | binomial    | n, p              |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Geometric(p)              | geom        | p                 |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Hypergeometric(N, n, K)   | hypergeom   | N, n, K           |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Poisson(lamda)            | poisson     | lamda             |     |         |
    +---------------------------+-------------+-------------------+-----+---------+

    Thus, each distribution above would have the same call signature::

        >>> import scipy.stats as ss
        >>> ss.your_dist_here(args, loc=loc, scale=scale)

    ANY SCIPY.STATS.DISTRIBUTION SHOULD WORK! IF ONE DOESN'T, PLEASE LET ME
    KNOW!

    Convenient constructors have been created to make assigning these
    distributions easier. They follow the parameter notation found in the
    respective Wikipedia articles:

    +---------------------------+---------------------------------------------------------------+
    | MCERP Distibution         | Wikipedia page                                                |
    +===========================+===============================================================+
    | N(mu, sigma)              | http://en.wikipedia.org/wiki/Normal_distribution              |
    +---------------------------+---------------------------------------------------------------+
    | U(a, b)                   | http://en.wikipedia.org/wiki/Uniform_distribution_(continuous)|
    +---------------------------+---------------------------------------------------------------+
    | Exp(lamda, [mu])          | http://en.wikipedia.org/wiki/Exponential_distribution         |
    +---------------------------+---------------------------------------------------------------+
    | Gamma(k, theta)           | http://en.wikipedia.org/wiki/Gamma_distribution               |
    +---------------------------+---------------------------------------------------------------+
    | Beta(alpha, beta, [a, b]) | http://en.wikipedia.org/wiki/Beta_distribution                |
    +---------------------------+---------------------------------------------------------------+
    | LogN(mu, sigma)           | http://en.wikipedia.org/wiki/Log-normal_distribution          |
    +---------------------------+---------------------------------------------------------------+
    | X2(df)                    | http://en.wikipedia.org/wiki/Chi-squared_distribution         |
    +---------------------------+---------------------------------------------------------------+
    | F(dfn, dfd)               | http://en.wikipedia.org/wiki/F-distribution                   |
    +---------------------------+---------------------------------------------------------------+
    | Tri(a, b, c)              | http://en.wikipedia.org/wiki/Triangular_distribution          |
    +---------------------------+---------------------------------------------------------------+
    | T(df)                     | http://en.wikipedia.org/wiki/Student's_t-distribution         |
    +---------------------------+---------------------------------------------------------------+
    | Weib(lamda, k)            | http://en.wikipedia.org/wiki/Weibull_distribution             |
    +---------------------------+---------------------------------------------------------------+
    | Bern(p)                   | http://en.wikipedia.org/wiki/Bernoulli_distribution           |
    +---------------------------+---------------------------------------------------------------+
    | B(n, p)                   | http://en.wikipedia.org/wiki/Binomial_distribution            |
    +---------------------------+---------------------------------------------------------------+
    | G(p)                      | http://en.wikipedia.org/wiki/Geometric_distribution           |
    +---------------------------+---------------------------------------------------------------+
    | H(M, n, N)                | http://en.wikipedia.org/wiki/Hypergeometric_distribution      |
    +---------------------------+---------------------------------------------------------------+
    | Pois(lamda)               | http://en.wikipedia.org/wiki/Poisson_distribution             |
    +---------------------------+---------------------------------------------------------------+

    Thus, the following are equivalent::

        >>> x = N(10, 1)
        >>> x = uv(ss.norm(loc=10, scale=1))

    Examples
    --------
    A three-part assembly

        >>> x1 = N(24, 1)
        >>> x2 = N(37, 4)
        >>> x3 = Exp(2)  # Exp(mu=0.5) works too

        >>> Z = (x1*x2**2)/(15*(1.5 + x3))
        >>> Z
        uv(1161.46231679, 116646.762981, 0.345533974771, 3.00791101068)

    The result shows the mean, variance, and standardized skewness and kurtosis
    of the output variable Z, which will vary from use to use due to the random
    nature of Monte Carlo simulation and latin-hypercube sampling techniques.

    Basic math operations may be applied to distributions, where all
    statistical calculations are performed using latin-hypercube enhanced Monte
    Carlo simulation. Nearly all of the built-in trigonometric-, logarithm-,
    etc. functions of the ``math`` module have uncertainty-compatible
    counterparts that should be used when possible since they support both
    scalar values and uncertain objects. These can be used after importing the
    ``umath`` module::

        >>> from mcerp.umath import * # sin(), sqrt(), etc.
        >>> sqrt(x1)
        uv(4.89791765647, 0.0104291897681, -0.0614940614672, 3.00264937735)

    At any time, the standardized statistics can be retrieved using::

        >>> x1.mean
        >>> x1.var  # x1.std (standard deviation) is also available
        >>> x1.skew
        >>> x1.kurt

    or all four together with::

        >>> x1.stats

    By default, the Monte Carlo simulation uses 10000 samples, but this can be
    changed at any time with::

        >>> mcerp.npts = number_of_samples

    Any value from 1,000 to 1,000,000 is recommended (more samples means more
    accurate, but also means more time required to perform the calculations).
    Although it can be changed, since variables retain their samples from one
    calculation to the next, this parameter should be changed before any
    calculations are performed to ensure parameter compatibility (this may
    change to be more dynamic in the future, but for now this is how it is).

    Also, to see the underlying distribution of the variable, and if matplotlib
    is installed, simply call its plot method::

        >>> x1.plot()

    Optional kwargs can be any valid kwarg used by matplotlib.pyplot.plot

    See Also
    --------
    N, U, Exp, Gamma, Beta, LogN, X2, F, Tri, PERT, T, Weib, Bern, B, G, H,
    Pois

    """

    def __init__(self, rv, tag=None):
        assert hasattr(rv, "dist"), "Input must be a distribution from the scipy.stats module."
        self.rv = rv
        self.tag = tag

        num_chunks = mp.cpu_count()
        chunk_size = npts // num_chunks
        if npts % num_chunks != 0:
            chunk_size += 1

        with mp.Pool(num_chunks) as pool:
            results = pool.map(self._generate_lhd_chunk, [chunk_size] * num_chunks)

        self._mcpts = np.concatenate(results)[:npts]

    def _generate_lhd_chunk(self, chunk_size):
        """
        Generate a chunk of Latin Hypercube samples for a given chunk_size.
        """
        np.random.seed()
        return lhd(dist=self.rv, size=chunk_size).flatten()


    def plot(self, hist=False, show=False, **kwargs):
        """
        Plot the distribution of the UncertainVariable. Continuous
        distributions are plotted with a line plot and discrete distributions
        are plotted with discrete circles.

        Optional
        --------
        hist : bool
            If true, a histogram is displayed
        show : bool
            If ``True``, the figure will be displayed after plotting the
            distribution. If ``False``, an explicit call to ``plt.show()`` is
            required to display the figure.
        kwargs : any valid matplotlib.pyplot.plot kwarg

        """
        import matplotlib.pyplot as plt

        if hist:
            vals = self._mcpts
            low = vals.min()
            high = vals.max()
            h = plt.hist(
                vals,
                bins=int(np.sqrt(len(vals)) + 0.5),
                histtype="stepfilled",
                density=True,
                **kwargs
            )
            plt.ylim(0, 1.1 * h[0].max())
        else:
            bound = 0.0001
            low = self.rv.ppf(bound)
            high = self.rv.ppf(1 - bound)
            if hasattr(self.rv.dist, "pmf"):
                low = int(low)
                high = int(high)
                vals = list(range(low, high + 1))
                plt.plot(vals, self.rv.pmf(vals), "o", **kwargs)
            else:
                vals = np.linspace(low, high, 500)
                plt.plot(vals, self.rv.pdf(vals), **kwargs)
        plt.xlim(low - (high - low) * 0.1, high + (high - low) * 0.1)

        if show:
            self.show()



uv = UncertainVariable  # a nicer form for the user

# DON'T MOVE THIS IMPORT!!! The prior definitions must be in place before
# importing the correlation-related functions
from .correlate import *
from . import umath
from . import stats


###############################################################################
# Define some convenience constructors for common statistical distributions.
# Hopefully these are a little easier/more intuitive to use than the
# scipy.stats.distributions.
###############################################################################

###############################################################################
# CONTINUOUS DISTRIBUTIONS
###############################################################################

def SkewNorm(alpha, loc, scale):
    return uv(ss.skewnorm(a=alpha, loc=loc, scale=scale))

def Beta(alpha, beta, low=0, high=1, tag=None):
    """
    A Beta random variate

    Parameters
    ----------
    alpha : scalar
        The first shape parameter
    beta : scalar
        The second shape parameter

    Optional
    --------
    low : scalar
        Lower bound of the distribution support (default=0)
    high : scalar
        Upper bound of the distribution support (default=1)
    """
    assert (
        alpha > 0 and beta > 0
    ), 'Beta "alpha" and "beta" parameters must be greater than zero'
    assert low < high, 'Beta "low" must be less than "high"'
    return uv(ss.beta(alpha, beta, loc=low, scale=high - low), tag=tag)


def BetaPrime(alpha, beta, tag=None):
    """
    A BetaPrime random variate

    Parameters
    ----------
    alpha : scalar
        The first shape parameter
    beta : scalar
        The second shape parameter

    """
    assert (
        alpha > 0 and beta > 0
    ), 'BetaPrime "alpha" and "beta" parameters must be greater than zero'
    x = Beta(alpha, beta, tag)
    return x / (1 - x)


def Bradford(q, low=0, high=1, tag=None):
    """
    A Bradford random variate

    Parameters
    ----------
    q : scalar
        The shape parameter
    low : scalar
        The lower bound of the distribution (default=0)
    high : scalar
        The upper bound of the distribution (default=1)
    """
    assert q > 0, 'Bradford "q" parameter must be greater than zero'
    assert low < high, 'Bradford "low" parameter must be less than "high"'
    return uv(ss.bradford(q, loc=low, scale=high - low), tag=tag)


def Burr(c, k, tag=None):
    """
    A Burr random variate

    Parameters
    ----------
    c : scalar
        The first shape parameter
    k : scalar
        The second shape parameter

    """
    assert c > 0 and k > 0, 'Burr "c" and "k" parameters must be greater than zero'
    return uv(ss.burr(c, k), tag=tag)


def ChiSquared(k, tag=None):
    """
    A Chi-Squared random variate

    Parameters
    ----------
    k : int
        The degrees of freedom of the distribution (must be greater than one)
    """
    assert int(k) == k and k >= 1, 'Chi-Squared "k" must be an integer greater than 0'
    return uv(ss.chi2(k), tag=tag)


Chi2 = ChiSquared  # for more concise use


def Erf(h, tag=None):
    """
    An Error Function random variate.

    This distribution is derived from a normal distribution by setting
    m = 0 and s = 1/(h*sqrt(2)), and thus is used in similar situations
    as the normal distribution.

    Parameters
    ----------
    h : scalar
        The scale parameter.
    """
    assert h > 0, 'Erf "h" must be greater than zero'
    return Normal(0, 1 / (h * 2 ** 0.5), tag)


def Erlang(k, lamda, tag=None):
    """
    An Erlang random variate.

    This distribution is the same as a Gamma(k, theta) distribution, but
    with the restriction that k must be a positive integer. This
    is provided for greater compatibility with other simulation tools, but
    provides no advantage over the Gamma distribution in its applications.

    Parameters
    ----------
    k : int
        The shape parameter (must be a positive integer)
    lamda : scalar
        The scale parameter (must be greater than zero)
    """
    assert int(k) == k and k > 0, 'Erlang "k" must be a positive integer'
    assert lamda > 0, 'Erlang "lamda" must be greater than zero'
    return Gamma(k, lamda, tag)


def Exponential(lamda, tag=None):
    """
    An Exponential random variate

    Parameters
    ----------
    lamda : scalar
        The inverse scale (as shown on Wikipedia). (FYI: mu = 1/lamda.)
    """
    assert lamda > 0, 'Exponential "lamda" must be greater than zero'
    return uv(ss.expon(scale=1.0 / lamda), tag=tag)


Exp = Exponential  # for more concise use


def ExtValueMax(mu, sigma, tag=None):
    """
    An Extreme Value Maximum random variate.

    Parameters
    ----------
    mu : scalar
        The location parameter
    sigma : scalar
        The scale parameter (must be greater than zero)
    """
    assert sigma > 0, 'ExtremeValueMax "sigma" must be greater than zero'
    p = U(0, 1)._mcpts[:]
    return UncertainFunction(mu - sigma * np.log(-np.log(p)), tag=tag)


EVMax = ExtValueMax  # for more concise use


def ExtValueMin(mu, sigma, tag=None):
    """
    An Extreme Value Minimum random variate.

    Parameters
    ----------
    mu : scalar
        The location parameter
    sigma : scalar
        The scale parameter (must be greater than zero)
    """
    assert sigma > 0, 'ExtremeValueMin "sigma" must be greater than zero'
    p = U(0, 1)._mcpts[:]
    return UncertainFunction(mu + sigma * np.log(-np.log(1 - p)), tag=tag)


EVMin = ExtValueMin  # for more concise use


def Fisher(d1, d2, tag=None):
    """
    An F (fisher) random variate

    Parameters
    ----------
    d1 : int
        Numerator degrees of freedom
    d2 : int
        Denominator degrees of freedom
    """
    assert (
        int(d1) == d1 and d1 >= 1
    ), 'Fisher (F) "d1" must be an integer greater than 0'
    assert (
        int(d2) == d2 and d2 >= 1
    ), 'Fisher (F) "d2" must be an integer greater than 0'
    return uv(ss.f(d1, d2), tag=tag)


F = Fisher  # for more concise use


def Gamma(k, theta, tag=None):
    """
    A Gamma random variate

    Parameters
    ----------
    k : scalar
        The shape parameter (must be positive and non-zero)
    theta : scalar
        The scale parameter (must be positive and non-zero)
    """
    assert (
        k > 0 and theta > 0
    ), 'Gamma "k" and "theta" parameters must be greater than zero'
    return uv(ss.gamma(k, scale=theta), tag=tag)



def LogNormal(mu, sigma, tag=None, version='canonical'):
    """
    A Log-Normal random variate

    Parameters
    ----------
    mu : scalar
        The location parameter
    sigma : scalar
        The scale parameter (must be positive and non-zero)
    version: `canonical` (default) | `old`
        The version of the method to run. The `old` method is the original.
        In the `canonical` method, the `mu` parameter is the mean and the `sigma` is
        the standard deviation *of the resulting lognormal distribution*.
        See https://stackoverflow.com/questions/50685839/am-i-doing-something-wrong-when-using-scipy-stats-when-implementing-a-lognormal for a fog of war covering `scipy.stats.lognorm` parameters.
    """
    if version == 'old':
        assert sigma > 0, 'Log-Normal "sigma" must be positive'
        return uv(ss.lognorm(sigma, loc=mu), tag=tag)
    elif version=='canonical':
        mean=mu
        sd=sigma
        mu = math.log(mean**2.0 / (sd**2.0 + mean**2.0)**0.5)
        sigma = (math.log(1.0 + sd**2.0/mean**2.0))**0.5
        return uv(ss.lognorm(s=sigma, loc=0, scale=np.exp(mu)), tag=tag)


LogN = LogNormal  # for more concise use


def Normal(mu, sigma, tag=None):
    """
    A Normal (or Gaussian) random variate

    Parameters
    ----------
    mu : scalar
        The mean value of the distribution
    sigma : scalar
        The standard deviation (must be positive and non-zero)
    """
    assert sigma > 0, 'Normal "sigma" must be greater than zero'
    return uv(ss.norm(loc=mu, scale=sigma), tag=tag)


N = Normal  # for more concise use


def Pareto(q, a, tag=None):
    """
    A Pareto random variate (first kind)

    Parameters
    ----------
    q : scalar
        The scale parameter
    a : scalar
        The shape parameter (the minimum possible value)
    """
    assert q > 0 and a > 0, 'Pareto "q" and "a" must be positive scalars'
    p = Uniform(0, 1, tag)
    return a * (1 - p) ** (-1.0 / q)


def Pareto2(q, b, tag=None):
    """
    A Pareto random variate (second kind). This form always starts at the
    origin.

    Parameters
    ----------
    q : scalar
        The scale parameter
    b : scalar
        The shape parameter
    """
    assert q > 0 and b > 0, 'Pareto2 "q" and "b" must be positive scalars'
    return Pareto(q, b, tag) - b


def PERT(low, peak, high, g=4.0, tag=None):
    """
    A PERT random variate

    Parameters
    ----------
    low : scalar
        Lower bound of the distribution support
    peak : scalar
        The location of the distribution's peak (low <= peak <= high)
    high : scalar
        Upper bound of the distribution support

    Optional
    --------
    g : scalar
        Controls the uncertainty of the distribution around the peak. Smaller
        values make the distribution flatter and more uncertain around the
        peak while larger values make it focused and less uncertain around
        the peak. (Default: 4)
    """
    a, b, c = [float(x) for x in [low, peak, high]]
    assert a <= b <= c, 'PERT "peak" must be greater than "low" and less than "high"'
    assert g >= 0, 'PERT "g" must be non-negative'
    mu = (a + g * b + c) / (g + 2)
    if mu == b:
        a1 = a2 = 3.0
    else:
        a1 = ((mu - a) * (2 * b - a - c)) / ((b - mu) * (c - a))
        a2 = a1 * (c - mu) / (mu - a)

    return Beta(a1, a2, a, c, tag)


def StudentT(v, tag=None):
    """
    A Student-T random variate

    Parameters
    ----------
    v : int
        The degrees of freedom of the distribution (must be greater than one)
    """
    assert int(v) == v and v >= 1, 'Student-T "v" must be an integer greater than 0'
    return uv(ss.t(v), tag=tag)


T = StudentT  # for more concise use


def Triangular(low, peak, high, tag=None):
    """
    A triangular random variate

    Parameters
    ----------
    low : scalar
        Lower bound of the distribution support
    peak : scalar
        The location of the triangle's peak (low <= peak <= high)
    high : scalar
        Upper bound of the distribution support
    """
    assert low <= peak <= high, 'Triangular "peak" must lie between "low" and "high"'
    low, peak, high = [float(x) for x in [low, peak, high]]
    return uv(
        ss.triang((1.0 * peak - low) / (high - low), loc=low, scale=(high - low)),
        tag=tag,
    )


Tri = Triangular  # for more concise use


def Uniform(low, high, tag=None):
    """
    A Uniform random variate

    Parameters
    ----------
    low : scalar
        Lower bound of the distribution support.
    high : scalar
        Upper bound of the distribution support.
    """
    assert low < high, 'Uniform "low" must be less than "high"'
    return uv(ss.uniform(loc=low, scale=high - low), tag=tag)


U = Uniform  # for more concise use


def Weibull(lamda, k, tag=None):
    """
    A Weibull random variate

    Parameters
    ----------
    lamda : scalar
        The scale parameter
    k : scalar
        The shape parameter
    """
    assert (
        lamda > 0 and k > 0
    ), 'Weibull "lamda" and "k" parameters must be greater than zero'
    return uv(ss.exponweib(lamda, k), tag=tag)


Weib = Weibull  # for more concise use


###############################################################################
# DISCRETE DISTRIBUTIONS
###############################################################################

def Integer(low, high, tag=None):
    """
    A random integer variable

    Parameters
    ----------
    low : int
        The low integer
    high : int
        The high integer
    """
    return uv(ss.randint(low, high+1), tag=tag)

Int = Integer


def Bernoulli(p, tag=None):
    """
    A Bernoulli random variate

    Parameters
    ----------
    p : scalar
        The probability of success
    """
    assert (
        0 < p < 1
    ), 'Bernoulli probability "p" must be between zero and one, non-inclusive'
    return uv(ss.bernoulli(p), tag=tag)


Bern = Bernoulli  # for more concise use


def Binomial(n, p, tag=None):
    """
    A Binomial random variate

    Parameters
    ----------
    n : int
        The number of trials
    p : scalar
        The probability of success
    """
    assert (
        int(n) == n and n > 0
    ), 'Binomial number of trials "n" must be an integer greater than zero'
    assert (
        0 < p < 1
    ), 'Binomial probability "p" must be between zero and one, non-inclusive'
    return uv(ss.binom(n, p), tag=tag)


B = Binomial  # for more concise use

def NegativeBinomial(n, p, tag=None):
    """
    A Negative Binomial random variate

    Parameters
    ----------
    n : int
        The number of trials
    p : scalar
        The probability of success
    """

    assert (
        int(n) == n and n > 0
    ), 'Negative Binomial number of trials "n" must be an integer greater than zero'
    assert (
        0 < p < 1
    ), 'Negative Binomial probability "p" must be between zero and one, non-inclusive'
    return uv(ss.nbinom(n,p), tag=tag)

def Polya(a,b, tag=None):
    """
    A Pólya random variate

    It is equal to `Poisson(Gamma(a, b)) = NegBin(a, 1/(1+b))`

    Parameters
    ----------
    a : int
        Parameter `a` to the `Gamma` distribution
    b : int | float
        Parameter `b` to the `Gamma` distribution
    """
    assert isinstance(a, int) and a > 0 , f"`a` should be an integer greater than zero, got:`{a}`"
    assert isinstance(b, (int, float)) and b > 0, f"`b` should be greater than zero, got `{b}`"
    return NegativeBinomial(a, 1/(1+b), tag=tag)


def Geometric(p, tag=None):
    """
    A Geometric random variate

    Parameters
    ----------
    p : scalar
        The probability of success
    """
    assert (
        0 < p < 1
    ), 'Geometric probability "p" must be between zero and one, non-inclusive'
    return uv(ss.geom(p), tag=tag)


G = Geometric  # for more concise use


def Hypergeometric(N, n, K, tag=None):
    """
    A Hypergeometric random variate

    Parameters
    ----------
    N : int
        The total population size
    n : int
        The number of individuals of interest in the population
    K : int
        The number of individuals that will be chosen from the population

    Example
    -------
    (Taken from the wikipedia page) Assume we have an urn with two types of
    marbles, 45 black ones and 5 white ones. Standing next to the urn, you
    close your eyes and draw 10 marbles without replacement. What is the
    probability that exactly 4 of the 10 are white?
    ::

        >>> black = 45
        >>> white = 5
        >>> draw = 10

        # Now we create the distribution
        >>> h = H(black + white, white, draw)

        # To check the probability, in this case, we can use the underlying
        #  scipy.stats object
        >>> h.rv.pmf(4)  # What is the probability that white count = 4?
        0.0039645830580151975

    """
    assert (
        int(N) == N and N > 0
    ), 'Hypergeometric total population size "N" must be an integer greater than zero.'
    assert (
        int(n) == n and 0 < n <= N
    ), 'Hypergeometric interest population size "n" must be an integer greater than zero and no more than the total population size.'
    assert (
        int(K) == K and 0 < K <= N
    ), 'Hypergeometric chosen population size "K" must be an integer greater than zero and no more than the total population size.'
    return uv(ss.hypergeom(N, n, K), tag=tag)


H = Hypergeometric  # for more concise use


def Poisson(lamda, tag=None):
    """
    A Poisson random variate

    Parameters
    ----------
    lamda : scalar
        The rate of an occurance within a specified interval of time or space.
    """
    assert lamda > 0, 'Poisson "lamda" must be greater than zero.'
    return uv(ss.poisson(lamda), tag=tag)


Pois = Poisson  # for more concise use


###############################################################################
# STATISTICAL FUNCTIONS
###############################################################################


def covariance_matrix(nums_with_uncert):
    """
    Calculate the covariance matrix of uncertain variables, oriented by the
    order of the inputs

    Parameters
    ----------
    nums_with_uncert : array-like
        A list of variables that have an associated uncertainty

    Returns
    -------
    cov_matrix : 2d-array-like
        A nested list containing covariance values

    Example
    -------

        >>> x = N(1, 0.1)
        >>> y = N(10, 0.1)
        >>> z = x + 2*y
        >>> covariance_matrix([x,y,z])
        [[  9.99694861e-03   2.54000840e-05   1.00477488e-02]
         [  2.54000840e-05   9.99823207e-03   2.00218642e-02]
         [  1.00477488e-02   2.00218642e-02   5.00914772e-02]]

    """
    ufuncs = list(map(to_uncertain_func, nums_with_uncert))
    cov_matrix = []
    for (i1, expr1) in enumerate(ufuncs):
        coefs_expr1 = []
        mean1 = expr1.mean
        for (i2, expr2) in enumerate(ufuncs[: i1 + 1]):
            mean2 = expr2.mean
            coef = np.mean((expr1._mcpts - mean1) * (expr2._mcpts - mean2))
            coefs_expr1.append(coef)
        cov_matrix.append(coefs_expr1)

    # Symmetrize the matrix:
    for (i, covariance_coefs) in enumerate(cov_matrix):
        covariance_coefs.extend(cov_matrix[j][i] for j in range(i + 1, len(cov_matrix)))

    return cov_matrix


def correlation_matrix(nums_with_uncert):
    """
    Calculate the correlation matrix of uncertain variables, oriented by the
    order of the inputs

    Parameters
    ----------
    nums_with_uncert : array-like
        A list of variables that have an associated uncertainty

    Returns
    -------
    corr_matrix : 2d-array-like
        A nested list containing covariance values

    Example
    -------

        >>> x = N(1, 0.1)
        >>> y = N(10, 0.1)
        >>> z = x + 2*y
        >>> correlation_matrix([x,y,z])
        [[ 0.99969486  0.00254001  0.4489385 ]
         [ 0.00254001  0.99982321  0.89458702]
         [ 0.4489385   0.89458702  1.        ]]

    """
    ufuncs = list(map(to_uncertain_func, nums_with_uncert))
    data = np.vstack([ufunc._mcpts for ufunc in ufuncs])
    return np.corrcoef(data.T, rowvar=0)

#
#
#

def mixture(variables: list[UncertainFunction], weights: list = None) -> UncertainFunction:
    """
    A mixture of random variables

    Parameters
    ----------
    variables : list(`UncertainFunction`)
        The original variables to get their mixture according to the given `weights`.
    weights : list(int)
        The weights of the mixture. Default is `None` which implies that the weight of each random variable is equal to 1.

    Returns
    -------
    out : `UncertainFunction`
        A random variable with the mixture of the given random variables
    """
    if not all([isinstance(var, UncertainFunction) for var in variables]):
        raise Exception(f"`variables` should all be of type `UncertainFunction`")
    variables=np.asarray(variables)

    if weights is None:
        weights=np.ones(len(variables), dtype=int)
    else:
        weights=np.asarray(weights)

    if not issubclass(weights.dtype.type, np.integer):
        raise Exception(f"`weights` must be integers")

    if len(weights) != len(variables):
        raise Exception("The length of the `variables` should equal the length of `weights`")

    out=np.empty(0)
    for i, w in zip(variables, weights):
        out=np.concatenate([out, np.tile(i,w)])

    drng = np.random.default_rng()
    return UncertainFunction(drng.choice(out, len(variables[0]), False))

def boxplot(variable,
            padding=0,
            title=None,
            xlabel=None,
            ylabel=None,
            fig=None,
            ax=None,
            figsize=(12, 5),
            fontsize=16,
            x_label_formatter=None,
            y_label_formatter=None,
            **kwargs
           ):

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    plot_kwargs = {
        'flierprops': {'markerfacecolor': 'black', 'marker': 'o', 'markersize': 3},
        'widths': 0.425,
    }
    plot_kwargs.update(kwargs)

    v=variable
    _ = ax.boxplot([v[x] for x in range(v.shape[0])][padding:], **plot_kwargs)

    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.grid(False)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.2, alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labeltop=False)
    ax.tick_params(axis='y', which='both', left=True, right=False, labelright=False)

    if x_label_formatter:
        ax.xaxis.set_major_formatter(FuncFormatter(x_label_formatter))
    if y_label_formatter:
        ax.yaxis.set_major_formatter(FuncFormatter(y_label_formatter))

    if title:
        ax.set_title(title, fontsize=fontsize+5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize+1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize+1)
    plt.tight_layout()