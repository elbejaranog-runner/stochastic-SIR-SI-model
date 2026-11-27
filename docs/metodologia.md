# Modelo y procedimiento

Los estados son proporciones humanas Sh, Ih, Rh y vectoriales Sv, Iv.
Wh y Wv son movimientos brownianos independientes.

```text
dSh = -βh Sh Iv dt - σh Sh Iv dWh
dIh = (βh Sh Iv - γ Ih)dt + σh Sh Iv dWh
dRh = γ Ih dt
dSv = (μv Iv - βv Sv Ih)dt - σv Sv Ih dWv
dIv = (βv Sv Ih - μv Iv)dt + σv Sv Ih dWv
Yk = ρ Nh ∫semana k βh Sh Iv dt
```

## Valores fijados

| Valor | Número | Procedencia |
|---|---|---|
| βh, βv, γ verdaderos | .35, .45, .10 por día | Diseño artificial |
| μv | .10 por día | Supuesto del piloto |
| ρ | .20 | Supuesto de notificación |
| X0 | (.98,.01,.01,.95,.05) | Estado artificial conocido |
| σh, σv | (.06,.075) o (.12,.15), día^-1/2 | Dos niveles experimentales |
| Nh | 4 658 811 personas | DANE, Valle del Cauca, 2022 |

El anexo DANE corresponde a proyecciones departamentales por área, revisión
8 de agosto de 2025, basada en CNPV 2018 actualizado. No se incorporaron
nacimientos o defunciones de estadísticas vitales. No se ha establecido que
no exista literatura pertinente: los valores restantes son supuestos no
calibrados con estos datos.

Fuente: https://www.dane.gov.co/files/censo2018/proyecciones-de-poblacion/Departamental/PPED-AreaDep-2018-2050_VP.xlsx

## Pseudoalgoritmos

### Generación
1. Fijar tasas verdaderas, X0 y semilla de réplica.
2. Integrar por Euler–Maruyama, paso .025 días.
3. Acumular la incidencia de cada semana. No añadir error de observación.

### TRF
1. Fijar un banco independiente de 32 trayectorias, paso .25 días.
2. Calcular incidencia media y sensibilidades respecto de las logtasas.
3. Minimizar `50*media(((predicho-observado)/max(observado))²)` con
   `scipy.optimize.least_squares(method="trf")` y cotas.
4. Guardar parámetros, curva y causa de parada.

### SGD
1. Usar el mismo banco y objetivo de TRF.
2. Barajar 26 semanas y formar **dos lotes de 13** por época.
3. Calcular el gradiente del lote, recortar su norma a 10 y actualizar logtasas
   con `eta=.0003/(1+paso/500)^.6`.
4. Proyectar a cotas. Reducir el paso a la mitad ante una salida del dominio,
   hasta 25 reducciones, sin recortar los estados.
5. Cada 20 pasos comprobar el gradiente completo: parar tras tres controles
   menores que 1e-5 o al alcanzar 4000 actualizaciones (2000 épocas).

### PINN
1. Crear red 1–32–32–5 con tanh y tres logtasas entrenables.
2. Imponer X0 y conservación mediante softmax humana y vectorial.
3. Integrar la incidencia por trapecios diarios.
4. Minimizar `L=100*Ld+.01*LEDE+100*Lrec` con Adam, 2000 actualizaciones,
   lr=.003, lotes de 13 semanas y 32 transiciones.
5. Continuar con L-BFGS de lote completo, hasta 2000 iteraciones.
6. Si un estado cae bajo 1e-12 o una varianza bajo 1e-30, interrumpir y
   restaurar el mejor punto factible. Informar la interrupción.

Ld es la mitad del error cuadrático normalizado de incidencia. LEDE es la
media de las transiciones gaussianas de Ih e Iv, incluyendo logvarianzas;
puede ser negativa. Lrec penaliza `ΔRh−γIhΔt`, con Δt=1 día.
Conservar las sumas de estados no garantiza que Rh sea creciente.

## Configuración reproducible

La configuración completa está en `results/raw/configuracion_congelada.json`.
TRF: máximo 400 evaluaciones; ftol=xtol=1e-9, gtol=1e-7.
Cotas TRF/SGD: βh,βv en [.02,1], γ en [.025,.5].
PINN usa tasas exponenciales sin cota superior: los dominios no son idénticos.
Adam: betas=(.9,.999), eps=1e-8, sin regularización ni clipping.
L-BFGS: lr=1, max_eval=2500, history_size=50, strong_wolfe,
tolerance_grad=1e-7, tolerance_change=1e-10.
La red utiliza float64. El tiempo de entrada se normaliza al horizonte.

## Lectura de las tablas

La réplica 1 se fija como ejemplo. Para cada método se elige entre los dos
inicios por pérdida de entrenamiento, sin consultar la verdad. Error relativo:
`abs(estimado-verdadero)/abs(verdadero)`, expresado como fracción.
El resumen de diez réplicas promedia errores absolutos por parámetro; no debe
confundirse con el error de una estimación media.

Son resultados retrospectivos, con X0, ρ, μv y σ conocidos por el estimador.
No prueban pronóstico, identificabilidad global ni validez epidemiológica.
Las dos condiciones están emparejadas y las inicializaciones no son réplicas
independientes. Los tiempos fueron medidos durante ejecución concurrente.
