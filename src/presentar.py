"""Regenera tablas breves, gráficas e informe con los ajustes guardados."""

from pathlib import Path
import argparse
import base64
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--entrada", type=Path, default=ROOT / "results/raw")
args = parser.parse_args()
r = pd.read_csv(args.entrada / "resultados.csv")
curvas = pd.read_csv(args.entrada / "curvas.csv")
hist = pd.read_csv(args.entrada / "historial.csv")
assert set(r.escenario) == {"bajo", "alto"}
assert not r.duplicated(["escenario", "replica", "metodo", "inicializacion"]).any()
r["criterio"] = np.where(r.metodo.eq("PINN"), r.loss, r.RMSE**2)
validos = r.dropna(subset=["criterio", "beta_h", "beta_v", "gamma"])
seleccion = validos.loc[
    validos.groupby(["escenario", "replica", "metodo"]).criterio.idxmin()
].copy()
TAB = ROOT / "results/tables"
FIG = ROOT / "results/figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
seleccion.to_csv(TAB / "seleccion_por_replica.csv", index=False)
verdad = {"beta_h": 0.35, "beta_v": 0.45, "gamma": 0.10}
simbolos = {"beta_h": "βh", "beta_v": "βv", "gamma": "γ"}
metodos = ["TRF", "SGD", "PINN"]
colores = ["#24738c", "#c77b24", "#8a4278"]
tablas = []
resumen = []
exportacion = {}
for nivel in ["bajo", "alto"]:
    exportacion[nivel] = []
    for metodo in metodos:
        g = seleccion[(seleccion.escenario == nivel) & (seleccion.metodo == metodo)]
        ejemplo = g[g.replica == 1].iloc[0]
        filas = []
        for p, v in verdad.items():
            fila = {
                "Parámetro": simbolos[p],
                "Verdadero": v,
                "Estimado": ejemplo[p],
                "Error relativo": abs(ejemplo[p] - v) / v,
            }
            filas.append(fila)
            tablas.append(dict(Nivel=nivel, Método=metodo, **fila))
            resumen.append(
                dict(
                    Nivel=nivel,
                    Método=metodo,
                    Parámetro=simbolos[p],
                    Réplicas=len(g),
                    Verdadero=v,
                    Estimado_medio=g[p].mean(),
                    Error_relativo_medio=np.mean(abs(g[p] - v) / v),
                )
            )
        exportacion[nivel].append(
            dict(
                metodo=metodo,
                estado=ejemplo.estado,
                inicializacion=int(ejemplo.inicializacion),
                filas=filas,
            )
        )
pd.DataFrame(tablas).to_csv(TAB / "tablas_parametros.csv", index=False)
pd.DataFrame(resumen).to_csv(TAB / "resumen_replicas.csv", index=False)
plt.rcParams.update(
    {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
)


def guardar(nombre):
    plt.savefig(FIG / (nombre + ".png"), dpi=170, bbox_inches="tight")
    plt.savefig(FIG / (nombre + ".svg"), bbox_inches="tight")
    plt.close()


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, nivel in zip(axes, ["bajo", "alto"]):
    for metodo, color in zip(metodos, colores):
        z = seleccion[
            (seleccion.escenario == nivel)
            & (seleccion.replica == 1)
            & (seleccion.metodo == metodo)
        ].iloc[0]
        q = curvas[
            (curvas.escenario == nivel)
            & (curvas.replica == 1)
            & (curvas.metodo == metodo)
            & (curvas.inicializacion == z.inicializacion)
        ]
        ax.plot(q.semana, q.ajustado, label=metodo, color=color)
    ax.scatter(q.semana, q.observado, c="black", s=18, label="Datos sintéticos")
    ax.set(title="Ruido " + nivel, xlabel="Semana", ylabel="Incidencia semanal")
    ax.legend(fontsize=9)
fig.tight_layout()
guardar("01_ajuste")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (p, v) in zip(axes, verdad.items()):
    for i, (metodo, color) in enumerate(zip(metodos, colores)):
        g = seleccion[seleccion.metodo == metodo]
        ax.scatter(
            np.full(len(g), i) + np.linspace(-0.15, 0.15, len(g)),
            g[p] / v,
            color=color,
            alpha=0.65,
        )
    ax.axhline(1, color="black", ls="--")
    ax.set_xticks(range(3), metodos)
    ax.set(
        title=simbolos[p] + " estimado / verdadero",
        yscale="log",
        ylabel="Razón, escala logarítmica",
    )
fig.tight_layout()
guardar("02_parametros")
fig, ax = plt.subplots(figsize=(6, 4))
for metodo, color in zip(metodos, colores):
    g = seleccion[seleccion.metodo == metodo]
    ax.scatter(g.RMSE, g.RMSE_reintegrado, label=metodo, color=color, alpha=0.7)
lo = min(seleccion.RMSE.min(), seleccion.RMSE_reintegrado.min())
hi = max(seleccion.RMSE.max(), seleccion.RMSE_reintegrado.max())
ax.plot([lo, hi], [lo, hi], "k--")
ax.set(
    xscale="log",
    yscale="log",
    xlabel="RMSE del ajuste",
    ylabel="RMSE al reintegrar tasas",
    title="Coherencia de los parámetros",
)
ax.legend()
guardar("03_coherencia")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metodo in zip(axes, ["SGD", "PINN"]):
    for nivel in ["bajo", "alto"]:
        z = seleccion[
            (seleccion.escenario == nivel)
            & (seleccion.replica == 1)
            & (seleccion.metodo == metodo)
        ].iloc[0]
        q = hist[
            (hist.escenario == nivel)
            & (hist.replica == 1)
            & (hist.metodo == metodo)
            & (hist.inicializacion == z.inicializacion)
        ]
        if metodo == "PINN":
            q = q[q.fase == "Adam"]
        ax.plot(q.paso, q.loss, label="Ruido " + nivel)
    ax.set(
        title=metodo,
        xlabel="Actualizaciones " + ("Adam" if metodo == "PINN" else "SGD"),
        ylabel="Pérdida completa",
    )
    ax.legend()
fig.tight_layout()
guardar("04_entrenamiento")


def imagen(nombre):
    return (
        '<img src="data:image/png;base64,'
        + base64.b64encode((FIG / (nombre + ".png")).read_bytes()).decode()
        + '">'
    )


def tabla(df):
    return df.to_html(index=False, float_format=lambda v: f"{v:.6f}", border=0)


html = """<!doctype html><html lang="es"><meta charset="utf-8"><title>Simulación SIR–SI</title>
<style>body{font:17px/1.6 Arial;max-width:1000px;margin:40px auto;padding:0 24px;color:#233747}h1,h2,h3{color:#18546a}table{border-collapse:collapse;margin:15px 0 30px;width:100%}th,td{padding:9px 14px;border-bottom:1px solid #ccd6df;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eaf1f5}img{max-width:100%}.nota{background:#fff3dc;padding:15px}code{background:#edf3f7}</style>
<h1>Simulaciones controladas SIR–SI</h1>
<p>Dos niveles de ruido del proceso, diez réplicas por nivel y dos inicializaciones por método: <b>120 ajustes</b>. Cada serie tiene 26 semanas continuas. Se observa la incidencia integrada sin añadir error de observación.</p>
<p>Se conservaron los resultados ya ejecutados de estos dos niveles, sin recalibrar. Las tasas verdaderas son βh=.35, βv=.45 y γ=.10 por día. Los niveles de ruido son (σh,σv)=(.06,.075) y (.12,.15), en día<sup>−1/2</sup>.</p>
<h2>Tablas de parámetros</h2><p>Formato de las tablas 2 y 3 de la referencia. Se muestra la <b>réplica 1</b>, fijada de antemano. La inicialización se selecciona por pérdida de entrenamiento. El error relativo es |estimado−verdadero|/|verdadero|: <b>0.10 equivale a 10 %</b>. Las estimaciones de las diez réplicas se conservan en los CSV.</p>"""
for nivel in ["bajo", "alto"]:
    html += "<h2>Ruido " + nivel + "</h2>"
    for t in exportacion[nivel]:
        html += f'<h3>{t["metodo"]}</h3>' + tabla(pd.DataFrame(t["filas"]))
        estado = (
            "Interrumpido; estimación del punto factible recuperado"
            if t["estado"] == "interrumpido"
            else (
                "Presupuesto agotado; sin convergencia"
                if t["metodo"] == "SGD"
                else "Criterio de parada del solver satisfecho"
            )
        )
        html += f'<p>Inicio {t["inicializacion"]}. {estado}.</p>'
html += "<h2>Ajuste de la incidencia</h2>" + imagen("01_ajuste")
html += "<h2>Recuperación de parámetros</h2>" + imagen("02_parametros")
html += "<p>Cada punto corresponde a una réplica y nivel, después de seleccionar el inicio. La línea de razón 1 representa recuperación exacta. Los niveles están emparejados; no son veinte realizaciones independientes.</p>"
html += (
    "<h2>Coherencia y entrenamiento</h2>"
    + imagen("03_coherencia")
    + imagen("04_entrenamiento")
)
html += """<p>Reintegrar las tasas con Euler–Maruyama comprueba si explican la curva aprendida. Las pérdidas SGD y PINN son distintas; sus valores no se comparan directamente. La pérdida PINN puede ser negativa por el logaritmo de las varianzas.</p>
<div class="nota"><b>Resultado:</b> TRF cumplió sus criterios de parada en 40/40 intentos. SGD agotó el presupuesto en 40/40. PINN se interrumpió en 40/40 y recuperó un punto factible. Sus curvas Rh seleccionadas decrecen en algún tramo, aunque el modelo exige dRh≥0. La configuración aún requiere revisión antes de interpretar parámetros de datos reales.</div>
<h2>Configuración esencial</h2>
<table><tr><th>Método</th><th>Configuración</th></tr>
<tr><td>TRF</td><td>32 trayectorias, paso .25 días, hasta 400 evaluaciones</td></tr>
<tr><td>SGD</td><td>4000 actualizaciones; 2 lotes de 13 semanas por época; tasa inicial .0003 decreciente</td></tr>
<tr><td>PINN</td><td>2 capas de 32 tanh; Adam 2000 pasos, lr=.003; L-BFGS hasta 2000 iteraciones</td></tr></table>
<p>Todos los hiperparámetros y pseudoalgoritmos están en <a href="metodologia.md">metodologia.md</a> y en el archivo de configuración. TRF/SGD tienen cotas de tasas; PINN no tiene cota superior. La comparación corresponde a estos procedimientos, no a optimizadores idénticos.</p>
<h2>Población y supuestos</h2>
<table><tr><th>Año</th><th>Población DANE: Valle del Cauca</th></tr><tr><td>2022</td><td>4 658 811</td></tr><tr><td>2023</td><td>4 673 395</td></tr><tr><td>2024</td><td>4 693 432</td></tr></table>
<p>Fuente: <a href="https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion">DANE, proyecciones departamentales</a>, revisión 8 agosto 2025. La simulación usa la población de 2022 como escala constante. μv=.10, ρ=.20, σ y X0=(.98,.01,.01,.95,.05) son supuestos experimentales conocidos, no valores del DANE.</p>
<p>Este estudio es de recuperación retrospectiva con parámetros fijos conocidos. No evalúa pronóstico ni valida el horizonte real de tres años. La aplicación continua a las 156 semanas de 2022–2024 está disponible en el <a href="informe_real.html">informe de datos reales</a>.</p>
<h2>Código para presentar</h2><p>Comenzar por <code>src/modelo_para_exponer.py</code>, seguir con <code>notebooks/01_simulacion.ipynb</code> y usar <code>src/simulacion.py</code> para reproducir el estudio. El motor detallado está en <code>src/metodos.py</code>.</p>
<p>Referencia de las tablas: Ogueda-Oliva et al. (2023), JMLMC 4(3), 17–35. Se adopta su forma de presentación; el modelo y el experimento aquí son distintos.</p></html>"""
(ROOT / "docs/informe.html").write_text(html, encoding="utf-8")
(TAB / "tablas_para_excel.json").write_text(
    json.dumps(exportacion, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("Listo: seis tablas de tres parámetros, cuatro figuras y el informe.")
