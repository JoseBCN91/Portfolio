import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import inspect
from helpers.stats import compare_distributions

# Constants
TRADING_DAYS_PER_WEEK = 5
MIN_TRADING_DAYS = 10
MOMENTUM_WINDOW = 3
MAX_TRADING_WEEKS = 6
PERCENTILE_CONVERSION = 100

# Setup logging
logger = logging.getLogger(__name__)


def analyze_month_patterns(month_num: int, data: pd.DataFrame) -> Tuple[Dict, Dict, Dict, List]:
    """Analyze trading patterns for a specific month.
    
    Analyzes weekly progression, momentum trends, and win rates based on
    historical data for a given calendar month.
    
    Args:
        month_num: Calendar month (1-12)
        data: DataFrame with columns: Month, Year, Daily_Return, Close, Day
        
    Returns:
        Tuple containing:
        - weekly_avg_returns: Dict[int, float] - Average returns (%) by trading week
        - momentum_avg: Dict[int, float] - Average momentum (%) by calendar day
        - win_rates: Dict[int, float] - Win rate (%) by calendar day
        - years_processed: List[int] - Years included in analysis
        
    Raises:
        ValueError: If month_num is not in range 1-12
        TypeError: If data is not a DataFrame
    """
    # Input validation
    if not isinstance(month_num, int) or month_num < 1 or month_num > 12:
        raise ValueError(f"month_num must be an integer between 1 and 12, got {month_num}")
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data)}")
    
    if data.empty:
        logger.warning(f"Empty DataFrame provided for month {month_num}")
        return {}, {}, {}, []

    # Filter data for the requested month
    month_data = data[data['Month'] == month_num].copy()
    
    if month_data.empty:
        logger.warning(f"No data available for month {month_num}")
        return {}, {}, {}, []

    # Initialize accumulators using defaultdict for efficiency
    from collections import defaultdict
    weekly_returns = defaultdict(list)
    momentum_by_day = defaultdict(list)
    positive_days_by_day = defaultdict(list)
    years_processed = []

    for year in sorted(month_data['Year'].unique()):
        year_month = month_data[month_data['Year'] == year].copy()

        # Filter valid data
        valid_mask = year_month['Daily_Return'].notna() & year_month['Close'].notna()
        year_month = year_month[valid_mask]

        if len(year_month) < MIN_TRADING_DAYS:
            logger.debug(f"Skipping {year}-{month_num}: only {len(year_month)} trading days (min {MIN_TRADING_DAYS})")
            continue

        years_processed.append(year)
        year_month = year_month.sort_index()

        # Calculate monthly returns
        first_price = year_month['Close'].iloc[0]
        year_month['Month_Return'] = (year_month['Close'] / first_price) - 1

        # Assign trading days and weeks
        year_month['Trading_Day'] = np.arange(1, len(year_month) + 1)
        year_month['Trading_Week'] = ((year_month['Trading_Day'] - 1) // TRADING_DAYS_PER_WEEK) + 1
        year_month['Day_of_Month'] = year_month.index.day

        # Weekly progression - vectorized operation
        for week in range(1, MAX_TRADING_WEEKS):
            week_data = year_month[year_month['Trading_Week'] == week]
            if len(week_data) > 0:
                week_return = week_data['Month_Return'].iloc[-1]
                weekly_returns[week].append(week_return)

        # Momentum analysis - 3-day rolling average
        if len(year_month) >= MOMENTUM_WINDOW:
            year_month['Rolling_Return'] = year_month['Daily_Return'].rolling(
                window=MOMENTUM_WINDOW, center=True
            ).mean()

            # Collect valid momentum values
            valid_momentum = year_month[year_month['Rolling_Return'].notna()]
            for day, momentum in zip(valid_momentum['Day_of_Month'], valid_momentum['Rolling_Return']):
                momentum_by_day[day].append(momentum)

        # Win rate analysis - vectorized
        valid_returns = year_month[year_month['Daily_Return'].notna()]
        for day, daily_return in zip(valid_returns['Day_of_Month'], valid_returns['Daily_Return']):
            positive_days_by_day[day].append(1 if daily_return > 0 else 0)

    # Calculate averages using numpy operations
    weekly_avg_returns = {
        week: np.mean(returns) * PERCENTILE_CONVERSION 
        for week, returns in weekly_returns.items()
    }
    momentum_avg = {
        day: np.mean(momentum) * PERCENTILE_CONVERSION 
        for day, momentum in momentum_by_day.items()
    }
    win_rates = {
        day: np.mean(positive) * PERCENTILE_CONVERSION 
        for day, positive in positive_days_by_day.items()
    }

    if years_processed:
        logger.info(f"Month {month_num} analysis: {len(years_processed)} years, "
                   f"{len(weekly_avg_returns)} weeks, {len(win_rates)} days")

    return weekly_avg_returns, momentum_avg, win_rates, years_processed


def get_month_summary(month_num: int, data: pd.DataFrame) -> dict:
    """Compute summary statistics for a given calendar month across years.

    Returns a dict with:
      - avg_monthly_return: mean of per-year month returns (in %)
      - median_monthly_return: median of per-year month returns (in %)
      - volatility: std dev of per-year month returns (in %)
      - positive_years: number of years with positive month return
      - years_processed: list of years included
    """
    if not isinstance(month_num, int) or month_num < 1 or month_num > 12:
        raise ValueError("month_num must be 1-12")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    month_data = data[data['Month'] == month_num]
    if month_data.empty:
        return {
            'avg_monthly_return': 0.0,
            'median_monthly_return': 0.0,
            'volatility': 0.0,
            'positive_years': 0,
            'years_processed': []
        }

    per_year_returns = []
    years = []
    for year in sorted(month_data['Year'].unique()):
        ym = month_data[month_data['Year'] == year].copy()
        ym = ym.dropna(subset=['Close'])
        if len(ym) < MIN_TRADING_DAYS:
            continue
        first_price = ym['Close'].iloc[0]
        last_price = ym['Close'].iloc[-1]
        month_return = (last_price / first_price) - 1
        per_year_returns.append(month_return)
        years.append(year)

    if not per_year_returns:
        return {
            'avg_monthly_return': 0.0,
            'median_monthly_return': 0.0,
            'volatility': 0.0,
            'positive_years': 0,
            'years_processed': []
        }

    arr = np.array(per_year_returns)
    avg = float(np.mean(arr) * 100)
    median = float(np.median(arr) * 100)
    vol = float(np.std(arr, ddof=1) * 100) if len(arr) > 1 else 0.0
    positive = int(np.sum(arr > 0))

    return {
        'avg_monthly_return': avg,
        'median_monthly_return': median,
        'volatility': vol,
        'positive_years': positive,
        'years_processed': years
    }


def compute_return_distribution(data: pd.DataFrame, period: str = 'daily', return_col: str = 'Daily_Return') -> tuple:
    """Compute return distribution series and summary statistics.

    Args:
        data: DataFrame with price/returns and Year/Month columns (index is datetime)
        period: 'daily' or 'monthly'
        return_col: column name for daily returns

    Returns:
        (returns_series, stats_dict)

    stats_dict contains: mean, median, std, skew, kurtosis (excess), jb (Jarque-Bera stat), jb_pvalue (if scipy available or None), n
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    # Prepare returns series
    if period == 'daily':
        if return_col not in data.columns:
            raise KeyError(f"Return column '{return_col}' not in data")
        returns = data[return_col].dropna().astype(float)
    elif period == 'monthly':
        # Compute monthly returns by first/last Close per Year-Month
        if 'Close' not in data.columns:
            raise KeyError("Data must include 'Close' column for monthly returns")
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            # attempt to convert index
            try:
                data = data.copy()
                data.index = pd.to_datetime(data.index)
            except Exception:
                raise TypeError("Data index must be datetime-like for monthly aggregation")

        monthly = data['Close'].resample('M')
        first = monthly.first()
        last = monthly.last()
        returns = (last / first - 1).dropna().astype(float)
    else:
        raise ValueError("period must be 'daily' or 'monthly'")

    n = int(returns.shape[0])
    stats = {
        'n': n,
        'mean': float(returns.mean()) if n else 0.0,
        'median': float(returns.median()) if n else 0.0,
        'std': float(returns.std(ddof=1)) if n > 1 else 0.0,
        'skew': float(returns.skew()) if n > 2 else 0.0,
        'kurtosis': float(returns.kurt()) if n > 3 else 0.0,  # excess kurtosis
        'jb': None,
        'jb_pvalue': None
    }

    # Compute Jarque-Bera statistic; p-value if scipy available
    try:
        S = stats['skew']
        K = stats['kurtosis']
        # JB = n*(S^2/6 + K^2/24)
        jb_stat = n * (S**2 / 6.0 + (K**2) / 24.0)
        stats['jb'] = float(jb_stat)
        try:
            from scipy.stats import chi2
            stats['jb_pvalue'] = float(1 - chi2.cdf(jb_stat, df=2))
        except Exception:
            stats['jb_pvalue'] = None
    except Exception:
        stats['jb'] = None
        stats['jb_pvalue'] = None

    return returns, stats


def select_distributions(familia='realline', verbose=False):
    """Select a subset of scipy.stats distributions based on domain.
    
    Parameters
    ----------
    familia : {'realline', 'realplus', 'realall'}
        realline: continuous distributions on domain (-inf, +inf) - best for financial returns
        realplus: continuous distributions on domain [0, +inf)
        realall: union of realline and realplus
        
    verbose : bool
        Whether to print distribution information
        
    Returns
    -------
    list : Selected distribution objects from scipy.stats
    """
    # moved to helpers.stats; keeping stub for backward compatibility if referenced
    from helpers.stats import select_distributions as _select
    return _select(familia=familia, verbose=verbose)


def compare_distributions(x, familia='realline', order_by='aic', verbose=False):
    """Fit and compare multiple distributions using AIC/BIC criteria.
    
    Parameters
    ----------
    x : array_like
        Data to fit distributions to
        
    familia : {'realline', 'realplus', 'realall'}
        Family of distributions to test
        
    order_by : {'aic', 'bic'}
        Criterion for ordering results
        
    verbose : bool
        Whether to print progress information
        
    Returns
    -------
    pd.DataFrame : Results with columns:
        - distribucion: distribution name
        - log_likelihood: log likelihood of fit
        - aic: Akaike Information Criterion
        - bic: Bayesian Information Criterion
        - n_parametros: number of parameters
        - parametros: fitted parameters dict
    """
    distribuciones = select_distributions(familia=familia, verbose=verbose)
    distribucion_ = []
    log_likelihood_= []
    aic_ = []
    bic_ = []
    n_parametros_ = []
    parametros_ = []
    
    x_array = np.asarray(x).flatten()
    n = len(x_array)
    
    logger.debug(f"Testing {len(distribuciones)} distributions on {n} data points")
    
    for i, distribucion in enumerate(distribuciones):
        try:
            parametros = distribucion.fit(data=x_array)
            
            # Get parameter names correctly: fit() returns [shape_params..., loc, scale]
            # The .shapes attribute tells us the shape parameter names
            if distribucion.shapes:
                shape_names = distribucion.shapes.split(',')
            else:
                shape_names = []
            
            # fit() returns shape parameters + loc + scale in that order
            nombre_parametros = shape_names + ['loc', 'scale']
            parametros_dict = dict(zip(nombre_parametros, parametros))
            
            # Calculate log likelihood using the fitted parameters
            log_likelihood = distribucion.logpdf(x_array, *parametros).sum()
            
            # Calculate AIC and BIC
            k = len(parametros)  # number of parameters
            aic = -2 * log_likelihood + 2 * k
            bic = -2 * log_likelihood + np.log(n) * k
            
            distribucion_.append(distribucion.name)
            log_likelihood_.append(log_likelihood)
            aic_.append(aic)
            bic_.append(bic)
            n_parametros_.append(k)
            parametros_.append(parametros_dict)
            
            if verbose:
                logger.info(f"  {distribucion.name}: AIC={aic:.2f}, BIC={bic:.2f}, params={k}")
                
        except Exception as e:
            logger.debug(f"Failed to fit {distribucion.name}: {str(e)[:100]}")
            
    logger.debug(f"Successfully fit {len(distribucion_)} distributions")
    
    resultados = pd.DataFrame({
                    'distribucion': distribucion_,
                    'log_likelihood': log_likelihood_,
                    'aic': aic_,
                    'bic': bic_,
                    'n_parametros': n_parametros_,
                    'parametros': parametros_,
                 })
    
    if len(resultados) > 0:
        resultados = resultados.sort_values(by=order_by).reset_index(drop=True)
        logger.debug(f"Top 3 fits:\n{resultados[['distribucion', 'aic', 'bic']].head(3)}")
    else:
        logger.warning("No distributions were successfully fit!")
    
    return resultados


def fit_distribution_to_returns(returns: 'pd.Series') -> dict:
    """Fit returns to best distribution using AIC criterion.

    Tests all continuous distributions on (-inf, +inf) domain and selects
    the best fit based on AIC (Akaike Information Criterion).

    Args:
        returns: pandas Series of returns (decimal, e.g., 0.01 for 1%)

    Returns:
        dict with keys:
          'best_dist': distribution name (str)
          'params': dict of fitted parameters
          'aic': AIC value
          'bic': BIC value
          'fitted_loc': location parameter
          'fitted_scale': scale parameter (if applicable)
          'fitted_shape': shape parameters as dict or None
    """
    r = returns.dropna().astype(float)
    if len(r) < 10:
        logger.warning("Too few samples for distribution fitting")
        return {
            'best_dist': 'normal',
            'params': {'loc': float(r.mean()), 'scale': float(r.std(ddof=1))},
            'aic': None,
            'bic': None,
            'fitted_loc': float(r.mean()),
            'fitted_scale': float(r.std(ddof=1)),
            'fitted_shape': None
        }

    try:
        # Compare all realline distributions using AIC
        results = compare_distributions(r, familia='realline', order_by='aic', verbose=False)
        
        if len(results) == 0:
            logger.warning("Failed to fit any distribution; using normal")
            mu, sigma = float(r.mean()), float(r.std(ddof=1))
            return {
                'best_dist': 'normal',
                'params': {'loc': mu, 'scale': sigma},
                'aic': None,
                'bic': None,
                'fitted_loc': mu,
                'fitted_scale': sigma,
                'fitted_shape': None
            }
        
        # Get best fit (first row, already sorted by AIC)
        best_row = results.iloc[0]
        best_dist_name = best_row['distribucion']
        best_params = best_row['parametros']
        best_aic = best_row['aic']
        best_bic = best_row['bic']
        
        # Extract loc and scale
        loc = float(best_params.get('loc', r.mean()))
        scale = float(best_params.get('scale', r.std(ddof=1)))
        
        # Extract shape parameters (everything except loc and scale)
        shape_params = {k: v for k, v in best_params.items() 
                       if k not in ['loc', 'scale']}
        
        logger.info(f"Distribution fitting: Best fit is '{best_dist_name}' with "
                   f"AIC={best_aic:.2f}, BIC={best_bic:.2f}")
        
        # Log top 5 fits for debugging
        top_fits = results.head(5)[['distribucion', 'aic', 'bic']].to_string()
        logger.debug(f"Top 5 distribution fits:\n{top_fits}")
        
        return {
            'best_dist': best_dist_name,
            'params': best_params,
            'aic': float(best_aic),
            'bic': float(best_bic),
            'fitted_loc': loc,
            'fitted_scale': scale,
            'fitted_shape': shape_params if shape_params else None
        }
        
    except Exception as e:
        logger.error(f"Error in distribution fitting: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        mu, sigma = float(r.mean()), float(r.std(ddof=1))
        return {
            'best_dist': 'normal',
            'params': {'loc': mu, 'scale': sigma},
            'aic': None,
            'bic': None,
            'fitted_loc': mu,
            'fitted_scale': sigma,
            'fitted_shape': None
        }

def get_distribution_stats(dist_info: dict) -> dict:
    from helpers.stats import get_distribution_stats as _stats
    return _stats(dist_info)

def get_goodness_of_fit(returns, dist_info):
    from helpers.stats import get_goodness_of_fit as _gof
    return _gof(returns, dist_info)
