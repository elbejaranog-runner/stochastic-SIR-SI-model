"""Versión legible del modelo, equivalente al EM acelerado de metodos.py.

El estudio usa Numba para repetir miles de integraciones. Esta versión NumPy
permite explicar el mismo paso numérico sin las sensibilidades del optimizador.
Las columnas son Sh, Ih, Rh, Sv, Iv y cada fila es una trayectoria.
"""

import numpy as np


def paso_euler_maruyama(x, tasas, mu, sigma, h, dW):
    beta_h, beta_v, gamma = tasas
    Sh, Ih, Rh, Sv, Iv = x.T
    infecciones = beta_h * Sh * Iv * h
    recuperaciones = gamma * Ih * h
    ruido_h = sigma[0] * Sh * Iv * dW[:, 0]
    cambio_vector = (beta_v * Sv * Ih - mu * Iv) * h
    ruido_v = sigma[1] * Sv * Ih * dW[:, 1]
    siguiente = np.column_stack(
        [
            Sh - infecciones - ruido_h,
            Ih + infecciones + ruido_h - recuperaciones,
            Rh + recuperaciones,
            Sv - cambio_vector - ruido_v,
            Iv + cambio_vector + ruido_v,
        ]
    )
    if not np.isfinite(siguiente).all() or siguiente.min() < -1e-12:
        raise FloatingPointError("EM salió del dominio; no se recortan estados")
    return siguiente, infecciones


def incidencia_semanal(tasas, x0, dW, N, rho=0.2, mu=0.1, sigma=(0.12, 0.15), h=0.25):
    pasos, trayectorias, _ = dW.shape
    pasos_semana = round(7 / h)
    assert np.isclose(pasos_semana * h, 7) and pasos % pasos_semana == 0
    x = np.tile(x0, (trayectorias, 1))
    casos = np.zeros(pasos // pasos_semana)
    for k in range(pasos):
        x, nuevas = paso_euler_maruyama(x, tasas, mu, sigma, h, dW[k])
        casos[k // pasos_semana] += rho * N * nuevas.mean()
    return casos


def perdida_datos(prediccion, observado):
    """Objetivo común de TRF y SGD en el estudio."""
    residuo = (prediccion - observado) / np.max(observado)
    return 50 * np.mean(residuo**2)
