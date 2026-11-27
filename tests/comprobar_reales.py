"""Comprueba continuidad, conservación, hiperparámetros y métricas reales."""

from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
raw = ROOT / "results/real/raw"
d = pd.read_csv(ROOT / "data/serie_semanal_real.csv")
r = pd.read_csv(raw / "resultados.csv")
p = pd.read_csv(raw / "predicciones.csv")
x = pd.read_csv(raw / "estados.csv")
c = json.loads((raw / "configuracion_real.json").read_text(encoding="utf-8"))
synthetic = json.loads(
    (ROOT / "results/raw/configuracion_congelada.json").read_text(encoding="utf-8")
)
assert len(d) == 156 and d.casos.sum() == 116309
assert pd.to_datetime(d.inicio).diff().dropna().eq(pd.Timedelta(days=7)).all()
assert len(r) == 6 and not r.duplicated(["metodo", "inicializacion"]).any()
assert (
    c["datos_sha256"]
    == hashlib.sha256((ROOT / "data/serie_semanal_real.csv").read_bytes()).hexdigest()
)
assert all(v == synthetic[k] for k, v in c["hiperparametros"].items())
for _, row in r.iterrows():
    q = p[(p.metodo == row.metodo) & (p.inicializacion == row.inicializacion)]
    if q.empty:
        continue  # Los fallos permanecen en resultados; no tienen predicción.
    assert len(q) == 156
    assert np.isclose(np.sqrt(np.mean((q.observado - q.predicho) ** 2)), row.RMSE)
    z = x[
        (x.metodo == row.metodo) & (x.inicializacion == row.inicializacion)
    ].sort_values("t_dias")
    assert z.t_dias.iloc[0] == 0 and z.t_dias.iloc[-1] == 1092
    assert not z.t_dias.duplicated().any()
    assert np.allclose(z[["Sh", "Ih", "Rh"]].sum(axis=1), 1, rtol=0, atol=1e-12)
    assert np.allclose(z[["Sv", "Iv"]].sum(axis=1), 1, rtol=0, atol=1e-12)
    assert np.allclose(
        z.iloc[0][["Sh", "Ih", "Rh", "Sv", "Iv"]].to_numpy(float),
        c["estados_iniciales"],
    )
    assert np.all(np.diff(z.t_dias) == (1 if row.metodo == "PINN" else 7))
print(
    "Verificados: seis ajustes, 156 semanas continuas, hiperparámetros iguales y métricas recalculadas."
)
