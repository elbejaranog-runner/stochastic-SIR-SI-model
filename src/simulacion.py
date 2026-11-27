"""Simulaciones controladas de incidencia semanal de dengue.

Ejecutar: python src/simulacion.py --replicas 10
Los hiperparámetros se conservan del ajuste conjunto, sin optimizarlos con
las réplicas de evaluación. Las dos inicializaciones NO son dos epidemias.
El motor detallado está en metodos.py; este archivo describe el experimento.
"""

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import metodos as m

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "raw"
VERDAD = np.array([0.35, 0.45, 0.1])
X0 = np.array([0.98, 0.01, 0.01, 0.95, 0.05])
INICIOS = [[0.15, 0.2, 1 / 7], [0.25, 0.12, 0.2]]
SEMANAS = 26
N = 4658811


def generar(semilla, sigma):
    """Una trayectoria EM independiente del banco usado para ajustar."""
    h = 0.025
    rng = np.random.default_rng(semilla)
    ruido = rng.normal(0, np.sqrt(h), (int(SEMANAS * 7 / h), 1, 2))
    escala = np.full(SEMANAS, m.C["rho"] * N)
    media, _, estados, valido, _, _ = m.simulate(
        np.log(VERDAD), X0, ruido, escala, h, m.C["mu"], *sigma
    )
    if not valido:
        raise FloatingPointError("Trayectoria generadora fuera del dominio")
    y = media.copy()
    datos = pd.DataFrame({"casos": y, "Nh": N})
    return (datos, media, estados)


def ejecutar(replicas, solo=None):
    config = dict(
        m.C,
        verdad=VERDAD.tolist(),
        X0=X0.tolist(),
        inicializaciones=INICIOS,
        semanas=SEMANAS,
        Nh=N,
        replicas_por_escenario=replicas,
        h_generador=0.025,
        protocolo="Hiperparámetros heredados y congelados; sin búsqueda",
    )
    raw = json.dumps(config, indent=2, ensure_ascii=False)
    anterior = OUT / "configuracion_congelada.json"
    if anterior.exists() and anterior.read_text(encoding="utf-8") != raw:
        raise ValueError("Configuración distinta: use otra carpeta con --salida")
    (OUT / "configuracion_congelada.json").write_text(raw, encoding="utf-8")
    fingerprint = hashlib.sha256(raw.encode()).hexdigest()

    def cargar(nombre):
        p = OUT / f"{nombre}.csv"
        return pd.read_csv(p).to_dict("records") if p.exists() else []

    filas, curvas = (cargar("resultados"), cargar("curvas"))
    historiales, generacion = (cargar("historial"), cargar("datos_sinteticos"))
    terminados = {
        (r["escenario"], r["replica"], r["inicializacion"], r["metodo"]) for r in filas
    }
    for nivel, sigma in [("bajo", (0.06, 0.075)), ("alto", (0.12, 0.15))]:
        m.C.update(sigma_h=sigma[0], sigma_v=sigma[1])
        escenario = nivel
        if solo and escenario != solo:
            continue
        for replica in range(1, replicas + 1):
            seed = 730000 + replica
            datos, media, verdad_x = generar(seed, sigma)
            if not any(
                (
                    g["escenario"] == escenario and g["replica"] == replica
                    for g in generacion
                )
            ):
                generacion.extend(
                    (
                        dict(
                            escenario=escenario,
                            replica=replica,
                            semana=k + 1,
                            observado=y,
                            incidencia_latente=z,
                            **{
                                name: v
                                for name, v in zip(
                                    ["Sh", "Ih", "Rh", "Sv", "Iv"], verdad_x[k + 1]
                                )
                            },
                        )
                        for k, (y, z) in enumerate(zip(datos.casos, media))
                    )
                )
            for j, inicio in enumerate(INICIOS, 1):
                for metodo in ["TRF", "SGD", "PINN"]:
                    if (escenario, replica, j, metodo) in terminados:
                        continue
                    key = dict(
                        escenario=escenario,
                        replica=replica,
                        inicializacion=j,
                        metodo=metodo,
                        semilla=seed,
                    )
                    fit_seed = 810000 + 10 * replica + j
                    if metodo == "PINN":
                        r, pred, estados, hist = m.fit_pinn(
                            datos,
                            X0,
                            inicio,
                            fit_seed,
                            f"{escenario}_{replica}_{j}",
                            np.arange(SEMANAS),
                            SEMANAS,
                        )
                    else:
                        r, pred, estados, hist = m.fit_direct(
                            metodo, datos, X0, inicio, fit_seed
                        )
                    r.update(key)
                    r["config_sha256"] = fingerprint
                    if pred is not None:
                        tasas = np.array([r[k] for k in ["beta_h", "beta_v", "gamma"]])
                        r["error_relativo_parametros_pct"] = float(
                            100 * np.mean(abs(tasas / VERDAD - 1))
                        )
                        r["RMSE_latente"] = float(np.sqrt(np.mean((pred - media) ** 2)))
                        r["min_incremento_Rh"] = float(np.diff(estados[:, 2]).min())
                        try:
                            em = m.Simulator(datos, X0)(np.log(tasas))[0]
                            r["RMSE_reintegrado"] = float(
                                np.sqrt(np.mean((em - datos.casos) ** 2))
                            )
                            r["reintegracion_valida"] = True
                        except FloatingPointError:
                            em = np.full(SEMANAS, np.nan)
                            r["reintegracion_valida"] = False
                        for k in range(SEMANAS):
                            curvas.append(
                                dict(
                                    **key,
                                    semana=k + 1,
                                    observado=datos.casos.iloc[k],
                                    latente=media[k],
                                    ajustado=pred[k],
                                    reintegrado=em[k],
                                )
                            )
                    filas.append(r)
                    historiales.extend((dict(**key, **h) for h in hist))
                    print(
                        "RESULTADO",
                        escenario,
                        replica,
                        j,
                        metodo,
                        r["estado"],
                        round(r.get("error_relativo_parametros_pct", float("nan")), 2),
                        flush=True,
                    )
                    for nombre, contenido in [
                        ("resultados", filas),
                        ("curvas", curvas),
                        ("historial", historiales),
                        ("datos_sinteticos", generacion),
                    ]:
                        pd.DataFrame(contenido).to_csv(
                            OUT / f"{nombre}.csv", index=False
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, default=10)
    parser.add_argument(
        "--solo", default=None, help="Una condición, para ejecución por partes"
    )
    parser.add_argument("--salida", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.salida.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    m.OUT = OUT.parent / "models"
    m.OUT.mkdir(parents=True, exist_ok=True)
    ejecutar(args.replicas, args.solo)
