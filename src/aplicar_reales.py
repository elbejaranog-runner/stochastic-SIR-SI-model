"""Ajusta una sola trayectoria a 2022–2024 con la configuración congelada."""

from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import metodos as m

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/real/raw"


def ejecutar():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/serie_semanal_real.csv"
    config = json.loads(
        (ROOT / "results/raw/configuracion_congelada.json").read_text(encoding="utf-8")
    )
    m.C.update({k: config[k] for k in m.C})
    data = pd.read_csv(source).sort_values(["ano", "semana"]).reset_index(drop=True)
    assert len(data) == 156
    assert pd.to_datetime(data.inicio).diff().dropna().eq(pd.Timedelta(days=7)).all()
    assert data.groupby("ano").size().eq(52).all()
    # Una única condición inicial en enero de 2022; supuesto, no estado observado.
    ih = data.casos.iloc[0] / (m.C["rho"] * data.Nh.iloc[0])
    iv = 0.2 * ih / (m.C["mu"] + 0.2 * ih)
    x0 = np.array([0.5, ih, 0.5 - ih, 1 - iv, iv])
    metadata = dict(
        hiperparametros=m.C,
        estados_iniciales=x0.tolist(),
        inicializaciones=config["inicializaciones"],
        semillas=[90101, 90102],
        semanas=156,
        dias=1092,
        sigma_referencia="alto",
        datos_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        interpretacion="Ajuste retrospectivo de las 156 semanas; sin reserva temporal",
    )
    (OUT / "configuracion_real.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    m.OUT = ROOT / "results/models/real"
    m.OUT.mkdir(parents=True, exist_ok=True)
    results = []
    predictions = []
    histories = []
    states = []
    for j, init in enumerate(config["inicializaciones"], 1):
        for method in ["TRF", "SGD", "PINN"]:
            print("INICIO", method, j, flush=True)
            if method == "PINN":
                row, pred, x, hist = m.fit_pinn(
                    data, x0, init, 90100 + j, "real_continuo", np.arange(156), 156
                )
            else:
                row, pred, x, hist = m.fit_direct(method, data, x0, init, 90100 + j)
            key = dict(metodo=method, inicializacion=j, semilla=90100 + j)
            if pred is not None:
                rates = np.array([row[k] for k in ["beta_h", "beta_v", "gamma"]])
                row["min_incremento_Rh"] = float(np.diff(x[:, 2]).min())
                row["error_conservacion"] = float(
                    max(abs(x[:, :3].sum(1) - 1).max(), abs(x[:, 3:].sum(1) - 1).max())
                )
                try:
                    em = m.Simulator(data, x0)(np.log(rates))[0]
                    row["RMSE_reintegrado"] = float(
                        np.sqrt(np.mean((em - data.casos.to_numpy()) ** 2))
                    )
                    row["reintegracion_valida"] = True
                except FloatingPointError:
                    em = np.full(156, np.nan)
                    row["reintegracion_valida"] = False
                for k in range(156):
                    predictions.append(
                        dict(
                            **key,
                            semana_continua=k + 1,
                            inicio=data.inicio.iloc[k],
                            ano=int(data.ano.iloc[k]),
                            observado=int(data.casos.iloc[k]),
                            predicho=pred[k],
                            reintegrado=em[k]
                        )
                    )
                dt = 1 if method == "PINN" else 7
                for k, values in enumerate(x):
                    states.append(
                        dict(
                            **key,
                            t_dias=k * dt,
                            **dict(zip(["Sh", "Ih", "Rh", "Sv", "Iv"], values))
                        )
                    )
            results.append(dict(**key, **row))
            histories.extend(dict(**key, **h) for h in hist)
            for name, rows in [
                ("resultados", results),
                ("predicciones", predictions),
                ("estados", states),
                ("historial", histories),
            ]:
                pd.DataFrame(rows).to_csv(OUT / (name + ".csv"), index=False)
            print("FIN", method, j, row["estado"], row.get("RMSE"), flush=True)


if __name__ == "__main__":
    ejecutar()
