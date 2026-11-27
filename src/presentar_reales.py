"""Presenta el ajuste retrospectivo continuo de los datos reales."""

from pathlib import Path
import base64, json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/real/raw"
TAB = ROOT / "results/real/tables"
FIG = ROOT / "results/real/figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
r = pd.read_csv(RAW / "resultados.csv")
assert len(r) == 6 and not r.duplicated(["metodo", "inicializacion"]).any()
p = pd.read_csv(RAW / "predicciones.csv")
x = pd.read_csv(RAW / "estados.csv")
data = pd.read_csv(ROOT / "data/serie_semanal_real.csv")
config = json.loads((RAW / "configuracion_real.json").read_text(encoding="utf-8"))
original = json.loads(
    (ROOT / "results/raw/configuracion_congelada.json").read_text(encoding="utf-8")
)
assert all(v == original[k] for k, v in config["hiperparametros"].items())
r["criterio"] = np.where(r.metodo.eq("PINN"), r.loss, r.RMSE**2)
validos = r.dropna(subset=["criterio", "beta_h", "beta_v", "gamma"])
s = (
    validos.loc[validos.groupby("metodo").criterio.idxmin()]
    .set_index("metodo")
    .reindex(["TRF", "SGD", "PINN"])
)
assert s.beta_h.notna().all()
parametros = pd.DataFrame(
    {
        "Parámetro": ["βh", "βv", "γ"],
        **{m: s.loc[m, ["beta_h", "beta_v", "gamma"]].to_numpy(float) for m in s.index},
    }
)
metricas = s.reset_index()[["metodo", "RMSE", "MAE", "R2"]].rename(
    columns={"metodo": "Método"}
)
fisica = pd.DataFrame(
    [
        {
            "Método": m,
            "Rh decrece": bool(s.loc[m, "min_incremento_Rh"] < -1e-10),
            "RMSE reintegrado": s.loc[m, "RMSE_reintegrado"],
            "Estado": s.loc[m, "estado"],
        }
        for m in s.index
    ]
)
anual = (
    data.groupby("ano")
    .agg(Casos=("casos", "sum"), Poblacion=("Nh", "first"))
    .reset_index()
    .rename(columns={"ano": "Año"})
)
paradas = r[
    ["metodo", "inicializacion", "estado", "convergencia", "terminacion", "segundos"]
].copy()
hiper = pd.DataFrame(
    [
        ["Serie", "156 semanas continuas; 1092 días; una sola condición inicial"],
        ["TRF", "32 trayectorias; h=.25 días; hasta 400 evaluaciones"],
        ["SGD", "Lotes de 13 semanas; 12 lotes por época; hasta 4000 actualizaciones"],
        ["SGD: tasa", ".0003/(1+paso/500)^.6; clipping=10"],
        ["PINN", "2×32 tanh; Adam 2000 pasos, lr=.003; L-BFGS hasta 2000 iteraciones"],
        [
            "PINN: lotes",
            "13 semanas; 32 transiciones; 166.67 épocas equivalentes de datos en Adam",
        ],
        ["Pérdida PINN", "100 Ld + .01 LEDE + 100 Lrec"],
        [
            "μv; ρ; σh; σv",
            ".10; .20; .12; .15 (supuestos; nivel alto del estudio controlado)",
        ],
        ["X0", str(config["estados_iniciales"])],
        [
            "Selección de inicio",
            "Menor criterio de entrenamiento entre dos inicios; no se usan valores verdaderos",
        ],
    ],
    columns=["Elemento", "Configuración"],
)
tables = {
    "Parametros": parametros,
    "Ajuste": metricas,
    "Coherencia": fisica,
    "Datos_anuales": anual,
    "Configuracion": hiper,
    "Paradas": paradas,
}
for name, t in tables.items():
    t.to_csv(TAB / (name + ".csv"), index=False)
s.reset_index().to_csv(TAB / "seleccion.csv", index=False)
(TAB / "tablas_excel.json").write_text(
    json.dumps(
        {
            k: v.astype(object).where(pd.notna(v), None).to_dict(orient="split")
            for k, v in tables.items()
        },
        ensure_ascii=False,
        allow_nan=False,
    ),
    encoding="utf-8",
)
colores = {"TRF": "#24738c", "SGD": "#c77b24", "PINN": "#8a4278"}
plt.rcParams.update(
    {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
)


def guardar(name):
    plt.savefig(FIG / (name + ".png"), dpi=170, bbox_inches="tight")
    plt.savefig(FIG / (name + ".svg"), bbox_inches="tight")
    plt.close()


def curva(m):
    return p[
        (p.metodo == m) & (p.inicializacion == s.loc[m, "inicializacion"])
    ].sort_values("semana_continua")


def eje(ax):
    ax.axvline(52.5, color="#999999", ls=":", lw=1)
    ax.axvline(104.5, color="#999999", ls=":", lw=1)
    ax.set_xticks([1, 26, 52, 78, 104, 130, 156])
    ax.set_xlabel("Semana continua: 2022 (1–52), 2023 (53–104), 2024 (105–156)")


fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(np.arange(1, 157), data.casos, color="black", label="Casos observados", lw=1.7)
for m, color in colores.items():
    q = curva(m)
    ax.plot(q.semana_continua, q.predicho, label=m, color=color)
eje(ax)
ax.set_ylabel("Casos semanales")
ax.set_title("Una trayectoria ajustada a 2022–2024")
ax.legend()
fig.tight_layout()
guardar("01_ajuste_continuo")
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ax, (m, color) in zip(axes, colores.items()):
    q = curva(m)
    ax.plot(q.semana_continua, q.observado - q.predicho, color=color)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Observado − ajustado")
    ax.set_title(m)
    eje(ax)
fig.tight_layout()
guardar("02_residuos")
fig, ax = plt.subplots(figsize=(12, 4.5))
q = curva("PINN")
ax.plot(q.semana_continua, q.observado, color="black", label="Observado")
ax.plot(q.semana_continua, q.predicho, color=colores["PINN"], label="Curva PINN")
ax.plot(
    q.semana_continua,
    q.reintegrado,
    color="#b23c3c",
    ls="--",
    label="EM con tasas PINN",
)
eje(ax)
ax.set_ylabel("Casos semanales")
ax.set_title("¿Las tasas PINN reproducen la curva aprendida?")
ax.legend()
fig.tight_layout()
guardar("03_reintegracion")
fig, ax = plt.subplots(figsize=(10, 4))
for m, color in colores.items():
    z = x[(x.metodo == m) & (x.inicializacion == s.loc[m, "inicializacion"])]
    ax.plot(z.t_dias / 7, z.Rh, label=m, color=color)
ax.set(
    xlabel="Semanas desde el inicio de 2022",
    ylabel="Fracción recuperada Rh",
    title="El modelo exige una fracción recuperada no decreciente",
)
ax.legend()
fig.tight_layout()
guardar("04_recuperados")


def table(t):
    return (
        '<div class="tabla">'
        + t.to_html(index=False, border=0, float_format=lambda v: f"{v:,.6g}")
        + "</div>"
    )


def image(n):
    return (
        '<img src="data:image/png;base64,'
        + base64.b64encode((FIG / (n + ".png")).read_bytes()).decode()
        + '">'
    )


html = (
    """<!doctype html><html lang="es"><meta charset="utf-8"><title>Dengue: datos reales 2022–2024</title>
<style>body{font:17px/1.6 Arial;max-width:1100px;margin:40px auto;padding:0 24px;color:#233747}h1,h2{color:#18546a}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:10px;border-bottom:1px solid #ccd6df;text-align:left}th{background:#eaf1f5}.tabla{overflow-x:auto;margin:20px 0}img{max-width:100%}.nota{background:#fff3dc;padding:18px}</style>
<p><a href="index.html">Proyecto completo</a> · <a href="informe.html">Simulaciones controladas</a></p>
<h1>Aplicación a los datos reales de 2022–2024</h1>
<p>Se ejecutaron TRF, SGD y PINN sobre las <b>156 semanas continuas</b>, con dos inicializaciones por método: seis ajustes nuevos. Se conservó la configuración de las simulaciones, usando el nivel de ruido alto (σh=.12, σv=.15). El horizonte es de 1092 días y las tasas son constantes durante los tres años.</p>
<p><b>Es un ajuste retrospectivo:</b> se utilizaron las 156 semanas para entrenar. No hay reserva temporal ni evaluación de pronóstico en esta ejecución. No se reinician estados al cambiar de año.</p>
<h2>Datos y población</h2>"""
    + table(anual)
    + """
<p>Se utilizan 116 309 registros agregados como casos semanales del evento 210, departamento de ocurrencia 76, incluidos Cali y Buenaventura; probables y confirmados, excluidos ajustes 6 y D. La serie se organiza por semana de inicio de síntomas. Los años corresponden a las etiquetas epidemiológicas del archivo; la semana 1 de 2024 comienza el 31 de diciembre de 2023.</p>
<p>Nh procede de las proyecciones departamentales DANE, revisión 8 agosto 2025. Cambia por año en la ecuación de observación: casos semanales = ρ Nh integral de βh Sh Iv. No se añadieron nacimientos o defunciones a las ecuaciones.</p>
<h2>Parámetros estimados</h2><p>Las tasas están en día<sup>−1</sup>. Se elige el inicio con menor criterio de entrenamiento: RMSE para TRF/SGD y pérdida total para PINN. En datos reales no conocemos los parámetros verdaderos; por tanto, <b>no se calcula error relativo paramétrico</b>.</p>"""
    + table(parametros)
    + """
<h2>Calidad del ajuste</h2><p>RMSE y MAE se expresan en casos semanales. R² es descriptivo dentro de la muestra, no capacidad predictiva fuera de ella.</p>"""
    + table(metricas)
    + image("01_ajuste_continuo")
    + image("02_residuos")
    + """
<h2>Coherencia del modelo</h2>"""
    + table(fisica)
    + image("03_reintegracion")
    + image("04_recuperados")
)
pin = s.loc["PINN"]
html += (
    f"""<div class="nota">La curva PINN tiene RMSE {pin.RMSE:,.2f}, pero al reintegrar sus tasas en Euler–Maruyama el RMSE es {pin.RMSE_reintegrado:,.2f}. Su mínimo incremento de Rh es {pin.min_incremento_Rh:.4g}. Un Rh decreciente contradice dRh=γIh dt≥0. Un buen ajuste visual no basta para validar estas tasas.</div>
<h2>Terminación de los seis ajustes</h2>"""
    + table(paradas)
    + """
<p>Una parada TRF por tolerancia no demuestra un óptimo global ni identificabilidad. Agotar el presupuesto SGD no equivale a converger. Una PINN interrumpida informa el punto factible recuperado, no una solución convergente. Las comparaciones están condicionadas a los supuestos y a los distintos objetivos y cotas de cada método.</p>
<h2>Hiperparámetros y estados iniciales</h2>"""
    + table(hiper)
    + """
<p>Se fijó Sh(0)=.5. Ih(0)=casos de la primera semana/(ρNh) es una aproximación, no una prevalencia observada. Iv(0)=.2 Ih(0)/(μv+.2 Ih(0)); el factor .2 también es supuesto. Rh(0)=1−Sh(0)−Ih(0) y Sv(0)=1−Iv(0). Estos estados se fijan solo al inicio de 2022. ρ, μv, σ y X0 no son estimaciones del DANE.</p>
<p>SGD: 4000 actualizaciones representan 333 épocas completas y cuatro lotes adicionales si agota el presupuesto. Adam mantiene 2000 actualizaciones, equivalentes a 166.67 recorridos de los datos, frente a 1000 en la simulación de 26 semanas. Conservar el presupuesto no conserva el número de épocas.</p>
<h2>Cómo reproducir</h2><pre>python src/aplicar_reales.py
python src/presentar_reales.py</pre>
<p>Datos agregados: data/serie_semanal_real.csv. Procedencia: data/procedencia_real.json. Resultados completos y configuración: results/real/raw/. Tablas y figuras: results/real/tables/ y results/real/figures/. El paquete no incluye registros individuales de pacientes.</p>
<p>Fuente demográfica: <a href="https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion">DANE, proyecciones de población</a>. Los casos provienen del archivo suministrado para este proyecto.</p></html>"""
)
(ROOT / "docs/informe_real.html").write_text(html, encoding="utf-8")
(ROOT / "docs/index.html").write_text(
    """<!doctype html><html lang="es"><meta charset="utf-8"><title>Dengue SIR–SI</title><style>body{font:19px/1.7 Arial;max-width:850px;margin:70px auto;color:#233747}a{color:#18546a}</style><h1>Dengue SIR–SI: simulación y datos reales</h1><p><a href="informe.html">1. Simulaciones controladas: tablas y gráficas</a></p><p><a href="informe_real.html">2. Aplicación continua a 2022–2024</a></p><p>Ambos estudios conservan los mismos hiperparámetros. Las limitaciones de convergencia y coherencia física están documentadas en cada informe.</p></html>""",
    encoding="utf-8",
)
print(parametros.to_string(index=False))
print(metricas.to_string(index=False))
print(fisica.to_string(index=False))
