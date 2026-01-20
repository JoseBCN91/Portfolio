import logging
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


def select_distributions(familia='realline', verbose=False):
    """Select a subset of scipy.stats distributions based on domain."""
    distribuciones = [getattr(sp_stats, d) for d in dir(sp_stats)
                     if isinstance(getattr(sp_stats, d), (sp_stats.rv_continuous, sp_stats.rv_discrete))]

    exclusiones = ['levy_stable', 'vonmises', 'studentized_range']
    distribuciones = [dist for dist in distribuciones if dist.name not in exclusiones]

    dominios = {
        'realline': [-np.inf, np.inf],
        'realplus': [0, np.inf],
        'realall' : [-np.inf, np.inf],
    }

    distribucion = []
    tipo = []
    dominio_inf = []
    dominio_sup = []

    for dist in distribuciones:
        distribucion.append(dist.name)
        tipo.append('continua' if isinstance(dist, sp_stats.rv_continuous) else 'discreta')
        dominio_inf.append(dist.a)
        dominio_sup.append(dist.b)

    info_distribuciones = pd.DataFrame({
        'distribucion': distribucion,
        'tipo': tipo,
        'dominio_inf': dominio_inf,
        'dominio_sup': dominio_sup
    })

    info_distribuciones = info_distribuciones.sort_values(by=['dominio_inf', 'dominio_sup']).reset_index(drop=True)

    if familia in ['realline', 'realplus', 'realall']:
        info_distribuciones = info_distribuciones[info_distribuciones['tipo']=='continua']
        condicion = (info_distribuciones['dominio_inf'] == dominios[familia][0]) & \
                    (info_distribuciones['dominio_sup'] == dominios[familia][1])
        info_distribuciones = info_distribuciones[condicion].reset_index(drop=True)

    seleccion = [dist for dist in distribuciones if dist.name in info_distribuciones['distribucion'].values]

    if verbose:
        logger.info(f"Selected {len(seleccion)} distributions for family '{familia}'")

    return seleccion


def compare_distributions(x, familia='realline', order_by='aic', verbose=False):
    """Fit and compare multiple distributions using AIC/BIC criteria."""
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

            if distribucion.shapes:
                shape_names = distribucion.shapes.split(',')
            else:
                shape_names = []

            nombre_parametros = shape_names + ['loc', 'scale']
            parametros_dict = dict(zip(nombre_parametros, parametros))

            log_likelihood = distribucion.logpdf(x_array, *parametros).sum()

            k = len(parametros)
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


def get_distribution_stats(dist_info: dict) -> dict:
    """Calculate statistics from fitted distribution."""
    try:
        dist_name = dist_info.get('best_dist', 'normal')
        params = dist_info.get('params', {})
        dist_obj = getattr(sp_stats, dist_name)

        shape_params = dist_info.get('fitted_shape', {}) or {}
        loc = dist_info.get('fitted_loc', 0)
        scale = dist_info.get('fitted_scale', 1)

        param_list = []
        if shape_params:
            for param_name in shape_params.keys():
                param_list.append(shape_params[param_name])
        param_list.extend([loc, scale])

        mean = float(dist_obj.mean(*param_list))
        std = float(dist_obj.std(*param_list))
        skew = float(dist_obj.stats(*param_list, moments='s'))
        kurtosis = float(dist_obj.stats(*param_list, moments='k'))

        return {
            'mean': mean,
            'std': std,
            'skew': skew,
            'kurtosis': kurtosis
        }
    except Exception as e:
        logger.debug(f"Could not calculate distribution stats: {e}")
        return {
            'mean': dist_info.get('fitted_loc', 0),
            'std': dist_info.get('fitted_scale', 1),
            'skew': 0,
            'kurtosis': 0
        }


def get_goodness_of_fit(returns, dist_info):
    """Calculate goodness-of-fit test (KS test) for fitted distribution."""
    try:
        from scipy.stats import kstest

        dist_name = dist_info.get('best_dist', 'normal')
        params = dist_info.get('params', {})
        dist_obj = getattr(sp_stats, dist_name)

        shape_params = dist_info.get('fitted_shape', {}) or {}
        loc = dist_info.get('fitted_loc', 0)
        scale = dist_info.get('fitted_scale', 1)

        param_dict = dict(shape_params) if shape_params else {}
        param_dict['loc'] = loc
        param_dict['scale'] = scale

        data = returns.dropna().astype(float).values

        ks_stat, ks_pval = kstest(data, lambda x: dist_obj.cdf(x, **param_dict))

        return {
            'ks_stat': float(ks_stat),
            'ks_pvalue': float(ks_pval)
        }
    except Exception as e:
        logger.debug(f'Could not calculate goodness-of-fit: {e}')
        return {
            'ks_stat': None,
            'ks_pvalue': None
        }
