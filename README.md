# Dengue SIR–SI: TRF, SGD y PINN

Simulaciones controladas y aplicación continua a los datos reales de 2022–2024
para comparar TRF, SGD y PINN. Incluye código Python, cuadernos y resultados.

**Estado del estudio:** se ejecutaron ambos experimentos. Las causas de parada
y la coherencia física se documentan en sus informes. Las estimaciones reales
son condicionales a los supuestos; un ajuste cercano no valida por sí solo las tasas.

## Ver los resultados

- [Entrada a los dos informes](docs/index.html).
- [Simulaciones controladas](docs/informe.html).
- [Aplicación real 2022–2024](docs/informe_real.html).
- [Parámetros estimados con datos reales](results/real/tables/Parametros.csv).
- [Tablas de parámetros](results/tables/tablas_parametros.csv).
- [Cuaderno paso a paso](notebooks/01_simulacion.ipynb).
- [Cuaderno de resultados reales](notebooks/02_datos_reales.ipynb).
- [Modelo NumPy para exponer](src/modelo_para_exponer.py).
- [Algoritmos e hiperparámetros](docs/metodologia.md).

Descargar el repositorio y abrir `docs/index.html` en el navegador.
GitHub permite leer el código y el cuaderno; el informe HTML se abre localmente.

## Organización

```text
src/                 Modelo, métodos, simulación y generación del informe
notebooks/           Simulación y lectura de resultados reales
data/                Población DANE y serie semanal real agregada
results/raw/         Los 120 ajustes y las series sintéticas
results/tables/      Tablas sencillas y resumen de las réplicas
results/figures/     Gráficas PNG y SVG
results/real/        Resultados, tablas y gráficas de 2022–2024
docs/                Informe, metodología y fuentes
tests/               Comprobación breve del modelo y los resultados
requirements.txt     Versiones de las librerías
```

## Instalación y ejecución

Python 3.12. Desde la carpeta del proyecto:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python tests/comprobar.py
python src/presentar.py
```

Los resultados ya están incluidos. Para ejecutar los 120 ajustes desde cero:

```bash
python src/simulacion.py --replicas 10 --salida results/nueva_ejecucion
python src/presentar.py --entrada results/nueva_ejecucion
```

Sin `--salida`, el programa reconoce los ajustes ya guardados y puede reanudar.
Los checkpoints nuevos se guardan en `results/models/`, excluidos de Git.
El Excel es una exportación de las tablas; los CSV y el informe se regeneran
con `presentar.py`. Para abrir el cuaderno instalar también `jupyterlab`.

## Experimento

Dos niveles de ruido del proceso: (σh,σv)=(.06,.075) y (.12,.15).
Cada nivel tiene 10 réplicas de 26 semanas, dos inicializaciones y tres métodos:
**120 ajustes**. Se observan las incidencias integradas del proceso sin añadir
error de observación. Los datos sintéticos pueden tener valores no enteros.
Las semillas se comparten entre niveles para comparaciones emparejadas.

Las tablas de cuatro columnas siguen la presentación de las tablas 2 y 3 de
Ogueda-Oliva et al. (2023): parámetro, verdadero, estimado y error relativo.
Muestran la réplica 1, fijada antes de comparar, con la inicialización elegida
por pérdida de entrenamiento. El error es una fracción: 0.10 equivale a 10 %.
El resumen de las diez réplicas se conserva por separado.

Los resultados proceden de los dos niveles ya ejecutados del estudio anterior.
Se reorganizaron sin recalibrar ni repetir los entrenamientos. No se ocultaron
interrupciones. Los hiperparámetros permanecen congelados.

## Datos reales y población

DANE aporta únicamente Nh: 4 658 811 (2022), 4 673 395 (2023), 4 693 432 (2024)
para Valle del Cauca. La simulación usa Nh de 2022 como escala constante.
ρ, μv, σ y X0 son supuestos experimentales, no estimaciones del DANE.
Fuentes y unidades en [metodología](docs/metodologia.md).

Se ejecutaron seis ajustes nuevos: dos inicializaciones por método sobre las
156 semanas continuas de 2022–2024, sin reiniciar estados en enero. Se conservaron
los hiperparámetros de las simulaciones y se fijó el ruido al nivel alto.
Se incluye exclusivamente la serie semanal agregada, no registros individuales.

```bash
python src/aplicar_reales.py
python src/presentar_reales.py
python tests/comprobar_reales.py
```

`aplicar_reales.py` vuelve a ejecutar los seis ajustes y reemplaza sus salidas.
`presentar_reales.py` regenera tablas CSV, figuras e informe a partir de ellas.
Los Excel entregados son instantáneas de las tablas. La aplicación real usa
todas las semanas para entrenar: es ajuste retrospectivo, no validación de
pronóstico. No hay valores verdaderos conocidos para calcular error paramétrico.

## Preparar la carga a GitHub

Crear un repositorio propio vacío y cargar el contenido de esta carpeta,
conservando sus subcarpetas. También puede usarse Git:

```bash
git init
git add .
git commit -m "Organiza simulaciones controladas SIR-SI"
git branch -M main
git remote add origin URL_DE_TU_REPOSITORIO
git push -u origin main
```

No se configuró un remoto ni se publicó contenido. Antes de publicar, completar
la autoría y elegir la licencia del trabajo; no se atribuye al usuario la
autoría del repositorio de referencia.

## Referencias

- Ogueda-Oliva et al. (2023). *Machine Learning for Predicting the Dynamics of
  Infectious Diseases during Travel through Physics Informed Neural Networks*.
  JMLMC 4(3), 17–35. [Repositorio indicado](https://github.com/aoguedao/dinn).
  Referencia metodológica; no se copió su código ni se reprodujo su SIRD.
- [DANE, proyecciones de población](https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion).
- [Dridi et al. (2021)](https://arxiv.org/abs/2105.08449): referencia para la
  aproximación de transición gaussiana; arquitectura distinta de la usada aquí.
