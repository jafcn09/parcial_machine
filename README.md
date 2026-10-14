# Examen Parcial · Redes Neuronales y Aprendizaje Profundo
### UNI · FIIS · 2026-II

Un solo notebook ejecutable —`parcial_redes_neuronales_e.ipynb`— resuelve las tres preguntas del
parcial. Cada pregunta escribe su propio paquete Python a disco con `%%writefile`, lo importa y lo
ejecuta: el notebook es a la vez informe y código fuente, y **corre de principio a fin sin descargar
un solo byte de datos reales** gracias a tres generadores de datos sintéticos (modo *demo*). Si los
datos reales están montados, el mismo notebook los usa sin cambiar una línea (modo *real*).

| # | Pregunta | Tarea | Datos reales | Fixture sintético |
|---|---|---|---|---|
| 1 | Rodilla (RSNA) | Clasificación multietiqueta 2.5D de 12 hallazgos con **etiquetas débiles extraídas de reportes en 12 idiomas** | `rsna-knee-abnormality-detection` (Kaggle) | `knee_ws/mock.py` → DICOM + reportes multilingües |
| 2 | Inundaciones (Sen1Floods11) | Segmentación de agua con **U-Net propia** sobre SAR Sentinel-1 (VV/VH) | `gs://sen1floods11` | `floods/mock.py` → GeoTIFF SAR con física de errores |
| 3 | Fauna (Caltech Camera Traps) | Detección de especies con **YOLOv8n** + módulo de monitoreo ecológico | LILA BC `eccv_18` | `wildlife/mock.py` → cámaras trampa sintéticas |

**Integrantes**

1. Cánepa Maceda, Jhafet Martín
2. Coaguila Berrios, Hipólito
3. Herrera Padilla, Lenin
4. Gutierrez Brenis, Berthing
5. Luque Canaza, Glenn
6. Mantari Martinez, Mario

---

## 1. Ejecución rápida

### Google Colab (recomendado · GPU T4)

```text
Runtime ▸ Change runtime type ▸ T4 GPU   →   Runtime ▸ Run all
```

No requiere configuración. El notebook detecta que no hay datos, genera los tres fixtures
sintéticos y ejecuta las 141 celdas de corrido: los 27 bloques cronometrados suman **~79 s** en
T4 (el notebook declara un presupuesto de 10 s por bloque y lo verifica al final, §22), más el
tiempo de instalación de dependencias y de generación de figuras.

### Local

```bash
git clone <este-repo> && cd parcial_machine

python -m venv .venv && source .venv/bin/activate      # Python 3.11+
pip install torch torchvision numpy pandas matplotlib scikit-learn \
            pydicom timm tifffile opencv-python "ultralytics>=8.2"

jupyter lab parcial_redes_neuronales_e.ipynb           # Run All
```

La primera celda instala por su cuenta lo que falte (`pydicom`, `timm`, `tifffile`), por lo que en
Colab/Kaggle basta con ejecutar. `ultralytics` se instala justo antes de la Pregunta 3.

> **Importante:** ejecutar en orden. Las celdas `%%writefile` crean los paquetes que las celdas
> siguientes importan; saltarse una deja el módulo sin escribir.

---

## 2. Los dos modos: demo y real

Cada pregunta decide su modo automáticamente al inicio de su bloque de configuración. No hay
que editar el notebook: se controla con variables de entorno.

```python
# Pregunta 1 (celda 30)
KNEE_MODE = os.environ.get("KNEE_MODE") or (
    "kaggle" if Path("/kaggle/input/rsna-knee-abnormality-detection").exists()
    else "cache" if Path("work/cache_224").exists()
    else "mock")
```

| Variable | Pregunta | Efecto |
|---|---|---|
| `KNEE_MODE` | 1 | Fuerza `kaggle` \| `cache` \| `mock` |
| `KNEE_DATA_ROOT` | 1 | Raíz alternativa del dataset RSNA |
| `FLOOD_DATA_ROOT` | 2 | Raíz de Sen1Floods11 (por defecto `data/sen1floods11`) |
| `CCT_DATA_ROOT` | 3 | Raíz de Caltech Camera Traps (por defecto `data/cct`) |

Lo que cambia entre modos es **escala, no lógica**: el mismo código, la misma pérdida, las mismas
particiones y las mismas métricas.

| | Pregunta 1 | Pregunta 2 | Pregunta 3 |
|---|---|---|---|
| **real** | 400 estudios + 60 holdout · 224 px · 8 cortes · ResNet-50 preentrenada · 12 épocas | 446 chips dorados · recorte 256 · U-Net 32×4 · 30 épocas | 3 000 fotos · 640 px · 30 épocas · YOLOv8n desde COCO |
| **demo** | 72 estudios sintéticos · 48 px · 3 cortes · ResNet-18 sin pesos · 3 épocas | 120 dorados + 240 débiles · 64 px · U-Net 16×3 · 5 épocas | 9 cámaras · 96 px · 3 épocas (~65 pasos) |

### Cómo apuntar a los datos reales

```bash
# Pregunta 1 · Kaggle: añadir el dataset al notebook; se detecta en /kaggle/input/...
export KNEE_DATA_ROOT=data/rsna-knee          # fuera de Kaggle

# Pregunta 2 · Sen1Floods11 (~1 GB la parte etiquetada a mano)
gsutil -m rsync -r gs://sen1floods11/v1.1/splits \
    data/sen1floods11/v1.1/splits
gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled \
    data/sen1floods11/v1.1/data/flood_events/HandLabeled
gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled \
    data/sen1floods11/v1.1/data/flood_events/WeaklyLabeled
export FLOOD_DATA_ROOT=data/sen1floods11

# Pregunta 3 · Caltech Camera Traps, desde https://lila.science/datasets/caltech-camera-traps
#   eccv_18_all_images_sm.tar.gz  +  CaltechCameraTrapsECCV18.json  +  eccv_18_annotation_files/
export CCT_DATA_ROOT=data/cct
```

---

## 3. Arquitectura del código

Tres paquetes independientes, un espacio de nombres por pregunta (`kplots`, `fplots`, `wplots`)
para que no colisionen dentro del mismo kernel.

```
knee_ws/                        # P1 · rodilla
├── config.py        Config (dataclass): único punto de verdad; cada experimento es un replace()
├── textlabels.py    extractor de etiquetas débiles en 12 idiomas (28 KB, el corazón de la P1)
├── dicom.py         selección de series, ventaneo percentil 1–99, caché .npz por estudio
├── data.py          dataset 2.5D (tripletas de cortes), sampler de clases raras
├── models.py        ScratchEncoder (1.3 M) · TimmEncoder · GatedAttentionPool (MIL)
├── losses.py        BCE ponderada por confianza · focal
├── train.py         bucle con early stopping sobre el holdout débil, umbrales por etiqueta
├── explain.py       Grad-CAM + fracción de energía en la región anatómica esperada
├── pipeline.py      particiones sin fuga (dorados nunca entrenan)
├── experiments.py   run_experiment / run_ablation
├── plots.py         figuras
└── mock.py          fixture DICOM + reportes sintéticos

floods/                         # P2 · inundaciones
├── config.py  data.py  mock.py
├── unet.py          U-Net propia, saltos concat | attention | none
├── losses.py        BCE · Dice · combo 0.5/0.5
├── metrics.py       IoU píxel/chip, F1, omisión, comisión, matriz de confusión
├── train.py  experiments.py
├── analysis.py      taxonomía de fallos SAR por retrodispersión VV
└── plots.py

wildlife/                       # P3 · fauna
├── config.py  data.py  mock.py
├── metrics.py       evaluador de detección propio (interpolación COCO de 101 puntos)
├── train.py         fine-tuning Ultralytics + inferencia
├── robustness.py    bins de tamaño y proxy de oclusión
├── monitor.py       eventos independientes (ventana 30 min) y reporte de actividad
├── explain.py       Grad-CAM sobre el backbone
└── plots.py
```

### Decisiones metodológicas que el código impone

Estas no son comentarios: el notebook las verifica con `assert` o en la tabla final de chequeos.

- **Ningún conjunto de prueba decide nada.** Los umbrales (por etiqueta en P1, de probabilidad en
  P2, de confianza en P3) se eligen en holdout/validación y se *aplican* al test.
- **P1 · el early stopping vigila el holdout débil, nunca el dorado.** El conjunto dorado queda
  como validación limpia y jamás entra al entrenamiento.
- **P2 · Bolivia está fuera de train, valid, test y débiles.** Es el conjunto fuera de distribución
  de los autores del dataset y solo se evalúa.
- **P3 · ninguna cámara `trans` aparece en entrenamiento.** La brecha cis/trans mide
  generalización a cámaras nuevas, no a fotos nuevas.
- **Ablaciones de un solo factor.** Cada fila difiere del baseline en exactamente una variable,
  vía `dataclasses.replace(cfg, ...)`.

---

## 4. Mapa de entregables

| Entregable | Sección | Qué produce |
|---|---|---|
| **Pregunta 1 · Rodilla** | | |
| 1 · etiquetas débiles | §3 | acuerdo, kappa, P/R/F1 y AUC por etiqueta contra el dorado |
| 2 · dos arquitecturas | §5 | CNN desde cero vs. transfer learning, misma cabeza y pérdida |
| 3 · evaluación dorada | §6 | AUC/P/R/F1 por etiqueta, umbral calibrado vs. 0.5 fijo |
| 4 · ablación | §7 | 8 configuraciones, un factor cada una |
| 5 · Grad-CAM | §8 | mapas + fracción de energía en LCA / menisco medial / derrame |
| 6 · desbalance | §4 | prevalencia por hallazgo + mitigaciones (sampler, focal, `pos_weight`) |
| 7 · sesgo lingüístico | §Entregable 7 | acuerdo y sobre-llamada del extractor por idioma |
| **Pregunta 2 · Inundaciones** | | |
| 1 · U-Net base | §10 | entrenada solo con chips dorados, BCE+Dice |
| 2 · ablación | §12 | pérdida · fuente de supervisión · diseño de saltos |
| 3 · evaluación test | §11 | IoU, F1, omisión/comisión, matriz, desglose por evento |
| 4 · generalización | §13 | brecha test vs. Bolivia, métrica a métrica |
| 5 · fallos SAR | §14 | sombra, suelo húmedo, doble rebote, borde — con ejemplos visuales |
| **Pregunta 3 · Fauna** | | |
| 1 · YOLO ajustado | §16 | decisiones de aumentación documentadas en tabla |
| 2 · evaluación `cis_test` | §17 | AP por especie + falsa alarma en fotos vacías |
| 3 · cis vs. trans | §18 | brecha + Grad-CAM como evidencia de memorización del fondo |
| 4 · robustez | §19 | AP/recall por tamaño de caja y tercil de oclusión |
| 5 · monitoreo ecológico | §20 | eventos independientes y reporte de actividad |
| 6 · uso responsable | §21 | discusión |
| **Verificación** | §22 | tabla de tiempos + 12 chequeos de integridad + `results.json` |

---

## 5. Ejemplos de salida

Salidas reales de la última ejecución completa en Colab T4, **modo demo** en las tres preguntas.

### Pregunta 1 · el extractor de etiquetas débiles (Entregable 1)

```text
[prepare] report column='Report' studies=72 gold=14 train=46 weak-holdout=12 planes=('Sagittal', 'Coronal')
[prepare] languages: {'en': 35, 'tr': 11, 'es': 11, 'de': 9, 'fr': 6}
⏱ P1 · prepare + etiquetas débiles: 1.05 s ✓

               label  n_gold  gold_pos  agreement  kappa  precision  recall     f1  auc_soft
0                ACL      14         4      0.786  0.571      0.571   1.000  0.727     0.850
2    Medial Meniscus      14         8      0.786  0.553      0.778   0.875  0.824     0.771
7           Effusion      14         6      0.857  0.696      1.000   0.667  0.800     0.833
12             MACRO      14        39      0.869  0.653      ...
```

`1 − agreement ≈ 0.13` es la tasa de ruido estimada de las etiquetas débiles: el número que
justifica ponderar la pérdida por confianza en lugar de tratar el target como verdad.

### Pregunta 1 · ablación de un solo factor (Entregable 4)

```text
                         name     arch   loss noise_strategy       planes  gold_macro_auc  gold_f1_macro
0            scratch_cnn_2p5d  scratch    bce       weighted  (Sagittal,)          0.7796         0.3330
1        transfer_bce_coronal     timm    bce       weighted   (Coronal,)          0.7018         0.2744
2     transfer_focal_weighted     timm  focal       weighted  (Sagittal,)          0.6505         0.3390
3       transfer_bce_weighted     timm    bce       weighted  (Sagittal,)          0.6409         0.4285
4  transfer_bce_no_oversample     timm    bce       weighted  (Sagittal,)          0.6399         0.4028
5        transfer_bce_shallow     timm    bce       weighted  (Sagittal,)          0.5331         0.3180
6       transfer_bce_highconf     timm    bce      high_conf  (Sagittal,)          0.4880         0.3523
7     transfer_bce_unweighted     timm    bce            all  (Sagittal,)          0.4616         0.2710
```

Lectura: la estrategia de ruido domina el ranking. `weighted` (0.641) supera a descartar celdas
dudosas (`high_conf`, 0.488) y a ignorar la confianza (`all`, 0.462) — con ~15 % de ruido, tirar
datos cuesta más que ponderarlos.

### Pregunta 2 · test en distribución (Entregable 3)

```text
test · IoU píxel=0.8136 | IoU chip=0.7538 | F1=0.8972 | P=0.8566 | R=0.9419 | omisión=0.0581 | comisión=0.1434

               pred: no agua  pred: agua
real: no agua          88023        3169
real: agua              1169       18935
```

### Pregunta 2 · brecha geográfica a Bolivia (Entregable 4)

```text
mejor configuración: gold+weak_combo

          metric  test_in_distribution  bolivia_held_out     gap
0      iou_pixel                0.8610            0.7459  0.1151
3      precision                0.8977            0.7666  0.1311
4         recall                0.9546            0.9650 -0.0104
6     commission                0.1023            0.2334 -0.1311
```

El patrón es el interesante: **el recall no se degrada, la precisión sí**. Fuera de distribución el
modelo no deja de ver el agua, sino que llama agua a superficies que no lo son (la comisión se
duplica) — consistente con suelo húmedo y sombras de radar con otra estadística de retrodispersión.

### Pregunta 3 · evidencia de memorización del fondo (Entregable 3)

```text
              metric     cis   trans     gap
4      animal_recall  0.0323  0.0303  0.0020
5  empty_false_alarm  0.0000  0.0638 -0.0638

      grupo  n  energia_en_caja  area_caja  ratio_energia_area  score_medio
0  no vista  4            0.087      0.076               1.164        0.011
1     vista  4            0.033      0.109               1.436        0.107
```

`ratio_energia_area ≈ 1` significa que el Grad-CAM es prácticamente uniforme: el mapa mira el
fondo, no el animal. En cámaras nuevas el ratio baja (1.16 vs. 1.44) y la falsa alarma en fotos
vacías pasa de 0 % a 6.4 %: dos señales independientes del mismo problema.

### Verificación final (§22)

```text
                                 verificación     ok
0           P1 dorados fuera de entrenamiento   True
1             P1 early stopping sobre holdout   True
2   P2 Bolivia fuera de train/valid/test/weak   True
3               P2 Bolivia solo en evaluación   True
4             P2 umbral elegido en validación   True
7     P3 cámaras trans fuera de entrenamiento   True
8   P3 umbral de confianza elegido en cis_val   True
9    P3 vacías como fondo en el entrenamiento   True
11                    todas las celdas ≤ 10 s  False

P1 scratch_cnn_2p5d: macro AUC dorado 0.7796 | P2 gold+weak_combo: IoU test 0.8610,
IoU Bolivia 0.7459 | P3 YOLOv8n: mAP@0.5 cis 0.0001, trans 0.0010 | máx. celda 23.46 s
```

---

## 6. Artefactos generados

```
work/                      # P1
├── figures/*.png           weak_label_agreement · prevalence · history_main · ablation · gradcam …
├── weak_labels.csv         target suave p_* y peso de confianza w_* por celda
├── weak_label_agreement.csv
├── ablation.csv
└── runs/<experimento>/     history.csv · metrics.csv · gold_report.csv

work_floods/               # P2
├── figures/*.png           history · threshold · confusion_test · gap_bolivia · error_examples …
├── ablation.csv
└── runs/<experimento>/

work_wildlife/             # P3
├── figures/*.png           species_counts · per_species_ap · generalization · robustness · activity …
└── runs/<experimento>/

results.json               # resumen combinado de las tres preguntas, para el informe
```

`results.json` contiene los modos usados, el mejor experimento de cada pregunta con sus métricas,
las tablas de ablación completas y el cronometraje de cada bloque.

---

## 7. Limitaciones conocidas (lectura honesta de los números)

Declaradas aquí porque un número fuera de contexto se malinterpreta:

1. **Las métricas de arriba son del modo demo, sobre datos sintéticos.** Miden que el pipeline
   es correcto y que las comparaciones son válidas entre sí, **no** el rendimiento clínico ni
   operativo de los modelos. Las magnitudes absolutas no son transferibles al dataset real.
2. **P3 · mAP ≈ 0.0001 es esperado y no es un fallo.** En demo YOLOv8n recibe ~65 pasos de
   gradiente a 96 px: suficiente para validar el pipeline de extremo a extremo, muy lejos de
   converger. Lo que sí es informativo en demo es el *patrón* relativo (brecha cis/trans, falsa
   alarma en vacías, caída monótona del recall con el tamaño de caja).
3. **Dos celdas exceden el presupuesto de 10 s:** `P3 · YOLO con vacías` (12.18 s) y
   `P3 · YOLO sin vacías` (23.46 s). El chequeo `todas las celdas ≤ 10 s` sale `False` por eso.
   Es el costo del lanzamiento del trainer de Ultralytics, que no se puede partir en celdas
   más pequeñas sin romper el entrenamiento; las 25 celdas restantes cumplen.
4. **El proxy de oclusión de la P3 es un estimador**, no una anotación: CCT no anota oclusión.
   En demo se audita contra la oclusión real del generador (Spearman) precisamente para
   justificar su uso en el modo real, donde no hay verdad disponible.
5. **El extractor multilingüe usa léxicos auditados a mano en 12 idiomas**, no un corpus anotado
   por idioma. §Entregable 7 reporta su acuerdo y su tasa de sobre-llamada por idioma para que el
   sesgo sea medible en lugar de implícito.

---

## 8. Reproducibilidad

- `seed_everything(seed)` fija `random`, `numpy` y `torch` en las tres preguntas; `cudnn.benchmark`
  se desactiva en demo para eliminar el autotuning no determinista.
- `warmup_device()` paga la inicialización del contexto CUDA **fuera** de las celdas cronometradas,
  para que los tiempos reportados sean del cómputo y no del arranque.
- Los fixtures sintéticos se generan con semilla fija (`seed=0`): la misma ejecución da los mismos
  números en la misma versión de PyTorch.
- `CELL_TIMES` y el contexto `timed()` registran cada bloque; §22 los imprime y los contrasta con
  el presupuesto.
