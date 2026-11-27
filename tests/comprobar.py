"""Prueba breve: reproduce datos y comprueba el motor NumPy pedagógico."""

from pathlib import Path
import sys, ast
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import metodos as m
from simulacion import generar, VERDAD, X0, N
from modelo_para_exponer import incidencia_semanal

r = pd.read_csv(ROOT / "results/raw/resultados.csv")
d = pd.read_csv(ROOT / "results/raw/datos_sinteticos.csv")
assert len(r) == 120 and len(d) == 520
assert set(r.escenario) == {"bajo", "alto"}
assert not r.duplicated(["escenario", "replica", "metodo", "inicializacion"]).any()
for nivel, sigma in [("bajo", (0.06, 0.075)), ("alto", (0.12, 0.15))]:
    for replica in range(1, 11):
        datos, _, _ = generar(730000 + replica, sigma)
        saved = d[(d.escenario == nivel) & (d.replica == replica)].sort_values("semana")
        assert np.allclose(datos.casos, saved.observado, rtol=1e-12, atol=1e-8)
sim = m.Simulator(datos, X0)
p, _, x = sim(np.log(VERDAD))
simple = incidencia_semanal(VERDAD, X0, sim.noise, N)
assert np.allclose(p, simple, rtol=1e-12, atol=1e-8)
assert np.allclose(x[:, :3].sum(1), 1, atol=1e-12, rtol=0)
assert np.allclose(x[:, 3:].sum(1), 1, atol=1e-12, rtol=0)
for p in (ROOT / "src").glob("*.py"):
    ast.parse(p.read_text(encoding="utf-8"))
print("Correcto: 120 ajustes, 20 series reproducidas y equivalencia del modelo NumPy.")
