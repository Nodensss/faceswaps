# Parity-тест этапов B–C: FaceFusion 3.7.1 и Android

Дата фиксации результатов: этап B — 2026-07-19, этап C — 2026-07-21.

## Область проверки

Этот набор проверяет три уровня из §11.4 `TECHNICAL_SPEC.md`:

1. геометрию — пять ключевых точек, affine matrix и выровненные кропы;
2. сырой выход swapper до mask compositing, inverse transform и blending.
3. финальный кадр после inverse transform, box-mask, feathering и blending.

Артефакты уровней 1–2 относятся к этапу B, уровня 3 — к этапу C. Все изображения
набора синтетические, созданы специально для теста и не
изображают пользователя или иных реальных людей.

## Зафиксированный десктопный эталон

| Компонент | Версия |
| --- | --- |
| FaceFusion | 3.7.1, commit `3f81a8a78454089d720b8f318a12ae1702c4633b` |
| Python | 3.12.10 |
| ONNX Runtime | 1.26.0 |
| OpenCV | 4.13.0 |
| NumPy | 2.2.1 |
| Execution Provider | `CPUExecutionProvider` |

Скрипт `run_facefusion_reference.py` импортирует из зафиксированного checkout и
вызывает production-функции FaceFusion `detect_with_yolo_face`,
`calculate_face_embedding`, `warp_face_by_face_landmark_5`, `prepare_crop_frame`,
`prepare_source_embedding`, `balance_source_embedding` и `forward_swap_face`.
Inference pools этих модулей инструментируются уже открытыми локальными ORT-
сессиями: это исключает скачивание и позволяет сохранить сырой tensor до маски и
обратной вставки. Вся нормализация и embedding conversion остаются внутри функций
FaceFusion; список вызванных функций записан в `desktop_results.json`.

Использованные веса лежали вне Git и до создания session проверялись скриптом по
полному SHA-256:

| Роль | Файл | Размер, байт | SHA-256 |
| --- | --- | ---: | --- |
| 5-point detector | `yoloface_8n.onnx` | 12 659 761 | `821cdbb1e65fbbabdde7dd0933f754797a343e56fd962729c61ffcefcd135929` |
| recognizer | `arcface_w600k_r50.onnx` | 174 388 474 | `f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1` |
| основной кандидат | `hyperswap_1a_256.onnx` | 402 742 682 | `c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add` |
| fallback | `inswapper_128_fp16.onnx` | 277 680 829 | `c4eccca86ad177586c85c28bf1a64a9d9ed237e283a15818d831f7facfd3f420` |

### Команда десктопного прогона

Команда выполнялась из корня репозитория в PowerShell. Пути указывают на локальный
checkout эталона, изолированное окружение и каталог весов на рабочей машине; ни один
из них не входит в репозиторий.

```powershell
$FaceFusionRoot = 'C:\Temp\facefusion-3.7.1'
$ModelDir = 'C:\Users\ozr\AppData\Local\FaceSwapLocal\models'
$ParityVenv = 'C:\Users\ozr\AppData\Local\FaceSwapLocal\facefusion-3.7.1-venv'

& "$ParityVenv\Scripts\python.exe" .\docs\parity\run_facefusion_reference.py `
  --facefusion-root $FaceFusionRoot `
  --model-dir $ModelDir `
  --input-root .\docs\parity\inputs `
  --output-root .\docs\parity\reference\facefusion-3.7.1 2>&1 |
  Tee-Object -FilePath .\docs\parity\reference\facefusion-3.7.1\reference-run.log
```

Сводка с версиями, хешами, landmarks, affine matrices, статистикой tensors и
временами сохранена в
`reference/facefusion-3.7.1/desktop_results.json`. Для каждой пары также сохранены
выровненные кропы, little-endian `float32` tensors и PNG-визуализации.

## Parity-набор

Каждый PNG имеет размер 1254×1254. В каждом изображении детектор выбирает одно лицо.

| Пара | SHA-256 source | SHA-256 target | Особенность цели |
| --- | --- | --- | --- |
| `pair_01` | `07d7dc1fdbfa61495342ba8e1b2178025c721561db9fd8e9dadaeb2b104ec3df` | `1e06b3cac962b9cb0cf9e3ecec3e963e4eccf62e1bfe26f9c5fc1c5b27f23f1c` | фронтальный ракурс |
| `pair_02` | `abe000ef539e541edd44fb83a94becc9fe5592b64c843a24b9fc96bcccb4faec` | `48f7c31dbade973344831db6007016fc3934651d1d4914ee732a62d25c6d5ad1` | очки и поворот |
| `pair_03` | `b2dfe8f885b0dcc01b13a4a5838b2e3d3c34d03a22af6bb9ffe6b7958328860f` | `4e29ad196ab9130356a5fe851bd479e15a8e145beb26dcd87a89305d13fab5c1` | борода, поворот и неоднородный свет |

## Android-прогон

Проверка выполнена на AVD `QuizMaster_API35`: Android 15 / API 35, x86_64,
`Android SDK built for x86_64`, 2 vCPU и 2 019 876 kB RAM. Это эмулятор, а не
reference device из §12. Все шесть inference-прогонов принудительно использовали
ONNX Runtime 1.26.0 CPU.

Предусловие: четыре файла весов импортированы через системный picker приложения;
`ModelStore` повторно проверил размер и SHA-256 в приватном каталоге приложения.
Тест прекращается до inference, если хотя бы одна модель не имеет статуса `Ready`.

```powershell
.\gradlew.bat assembleDebug assembleDebugAndroidTest --console=plain
adb install -r .\app\build\outputs\apk\debug\app-debug.apk
adb install -r .\app\build\outputs\apk\androidTest\debug\app-debug-androidTest.apk
adb shell am instrument -w -r `
  -e class com.faceswaplocal.app.inference.FaceFusionParityInstrumentedTest `
  com.faceswaplocal.app.test/androidx.test.runner.AndroidJUnitRunner
```

Результат финального instrumentation: `OK (1 test)`, 201,851 s. Один test method выполняет
3 пары × 2 swapper-модели последовательно. Полученные metrics и PNG скопированы в
`android/api35-x86_64/`.

## Метод сравнения и допуски

- landmarks: максимум абсолютной ошибки по x/y для любой из пяти точек — не более
  2 px;
- source/target aligned crop: SSIM не ниже 0,95;
- сырой tensor swapper: SSIM не ниже 0,95;
- backend detector, recognizer и swapper в этом тесте обязан быть `CPU`.

SSIM вычисляется непосредственно по `float32` tensors, а не по сжатым PNG:
7×7 uniform window, три RGB-канала, `data_range = 1`, `C1 = 0,01²`,
`C2 = 0,03²`, sample covariance. Выход HyperSwap переводится из `[-1; 1]` в
`[0; 1]`; выход InSwapper уже находится в `[0; 1]`. Реализация находится в
`FaceFusionParityInstrumentedTest`.

## Результаты

`matrix error` — максимальная абсолютная ошибка среди коэффициентов source/target
affine matrices для данной строки.

| Пара | Swapper | Landmarks max, px | Matrix error max | Source crop SSIM | Target crop SSIM | Raw output SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `pair_01` | HyperSwap 1a 256 | 0,051795 | 0,044586 | 0,998911 | 0,997928 | 0,998060 |
| `pair_01` | InSwapper 128 fp16 | 0,051795 | 0,027600 | 0,998911 | 0,998517 | 0,998275 |
| `pair_02` | HyperSwap 1a 256 | 0,054703 | 0,034561 | 0,998367 | 0,997617 | 0,997760 |
| `pair_02` | InSwapper 128 fp16 | 0,054703 | 0,032110 | 0,998367 | 0,998371 | 0,997551 |
| `pair_03` | HyperSwap 1a 256 | 0,123990 | 0,064592 | 0,998169 | 0,997606 | 0,997513 |
| `pair_03` | InSwapper 128 fp16 | 0,123990 | 0,032296 | 0,998169 | 0,998308 | 0,997645 |

Итог по численным порогам:

- максимальная ошибка landmarks — 0,123990 px;
- минимальный SSIM source crop — 0,998169;
- минимальный SSIM target crop — 0,997606;
- минимальный SSIM сырого выхода — 0,997513;
- все значения лучше обязательных допусков §11.4.

Визуальная проверка квадратных кропов дала разный качественный вывод при одинаково
успешной численной parity: InSwapper узнаваемо переносит личность source во всех трёх
парах, а HyperSwap на этом синтетическом наборе сохраняет слишком много признаков
target. Поскольку Android-выход HyperSwap совпадает с десктопным, это не признак
ошибки мобильной нормализации или channel order. До проверки на расширенном
лицензированном наборе InSwapper выбран активным fallback и UI-default, а HyperSwap
сохранён как основной кандидат shortlist.

## Документированное отклонение от полного FaceFusion-пайплайна

FaceFusion 3.7.1 в полном production flow может уточнять исходные пять точек через
дополнительный 68-point landmarker. В parity этапа B обе стороны намеренно используют
одни и те же исходные пять точек `yoloface_8n`; отдельный landmarker в текущий набор
моделей не включён. Это изолирует и доказывает parity уже реализованной цепочки
`YOLO 5 points → alignment → ArcFace → swapper`, но не является сравнением с
уточнённой 68→5 геометрией полного CLI. ML Kit применяется только для быстрого UI
preview и никогда не передаёт свои точки в neural pipeline.

Причина отклонения: на этапе B требуется 5-point detector; 68/106-point landmarker
нужен позднее для улучшенной маски/occlusion flow. Перед включением такого landmarker
для него потребуется отдельная лицензия, MODEL_CARD и parity геометрии.

### Android-реализация affine и resampling

Числовые шаблоны и порядок операций совпадают с FaceFusion, но две низкоуровневые
реализации намеренно отличаются:

- FaceFusion оценивает similarity matrix через
  `cv2.estimateAffinePartial2D(..., method=RANSAC, ransacReprojThreshold=100)`;
  Android использует детерминированное closed-form least-squares решение по всем пяти
  соответствиям. Причина — отсутствие зависимости от OpenCV в Android-приложении и
  воспроизводимость результата без случайного RANSAC. На текущих fixtures максимальная
  ошибка коэффициента matrix — 0,064592, а отдельные unit-тесты совпадают с шестью
  FaceFusion matrices с допуском `1e-5`, когда входные landmarks одинаковы.
- FaceFusion вызывает `cv2.warpAffine` с `INTER_AREA` и `BORDER_REPLICATE`; Android
  выполняет собственную bilinear inverse sampling с ограничением координат краем.
  Аналогичный bilinear resize применяется перед YOLO вместо OpenCV resize. Причина —
  детерминированный Android-only путь без нативного OpenCV. Измеренный эффект включён
  в parity: ошибка landmarks не выше 0,123990 px, SSIM выровненных кропов не ниже
  0,997606 и сырого выхода не ниже 0,997513.

Эти отклонения не считаются незаметными по умолчанию: thresholds instrumentation-
теста остаются обязательным регрессионным барьером при расширении набора ракурсов.

## Карта артефактов

- `inputs/` — шесть синтетических исходных PNG;
- `run_facefusion_reference.py` — воспроизводимый desktop reference runner;
- `reference/facefusion-3.7.1/desktop_results.json` — сводка эталона;
- `reference/facefusion-3.7.1/pair_*/metadata.json` — геометрия и статистика пары;
- `reference/facefusion-3.7.1/pair_*/*_f32le.bin` — эталонные float tensors;
- `android/api35-x86_64/android_results.json` — Android metrics;
- `android/api35-x86_64/*.png` — Android-визуализации сырого выхода.

## Parity этапа C: финальный кадр

Эталон создан тем же runner и окружением: FaceFusion 3.7.1, commit
`3f81a8a78454089d720b8f318a12ae1702c4633b`, Python 3.12.10, ONNX Runtime 1.26.0,
OpenCV 4.13.0, NumPy 2.2.1, `CPUExecutionProvider`. Production
`face_swapper.swap_face` запущен с `inswapper_128_fp16`, pixel boost `128x128`,
weight `0.5`, mask types `[box]`, blur `0.3`, padding `(0, 0, 0, 0)`.

FaceFusion 3.7.1 в этом production-пути не выполняет отдельное color matching.
Android намеренно добавляет требуемое FR-PHOTO-06 masked RGB mean/std color matching
с strength `0.65`. Поэтому побитовое совпадение финального PNG не ожидается; это
документированное функциональное отличие, а не ошибка alignment/swapper. Точные
per-pair gain/offset записаны в Android `stage_c_results.json`.

```powershell
# Desktop reference; команда обновляет артефакты этапов B и C
& 'C:\Users\ozr\AppData\Local\FaceSwapLocal\facefusion-3.7.1-venv\Scripts\python.exe' `
  .\docs\parity\run_facefusion_reference.py `
  --facefusion-root 'C:\Temp\facefusion-3.7.1' `
  --model-dir 'C:\Users\ozr\AppData\Local\FaceSwapLocal\models' `
  --input-root .\docs\parity\inputs `
  --output-root .\docs\parity\reference\facefusion-3.7.1

# Android API 35 / x86_64 / ONNX Runtime Android 1.26.0 / CPU
.\gradlew.bat assembleDebug assembleDebugAndroidTest --console=plain
adb install -r .\app\build\outputs\apk\debug\app-debug.apk
adb install -r .\app\build\outputs\apk\androidTest\debug\app-debug-androidTest.apk
adb shell am instrument -w -r `
  -e class com.faceswaplocal.app.inference.FaceFusionFinalFrameParityInstrumentedTest `
  com.faceswaplocal.app.test/androidx.test.runner.AndroidJUnitRunner
```

Метрики рассчитаны по RGB-пикселям Android PNG и canonical FaceFusion PNG. `MAE`
указан в уровнях `[0; 255]`; ROI — ограничивающий прямоугольник inverse paste-back.
Во всех парах вне ROI изменено ровно 0 пикселей.

| Пара | Full-frame global SSIM | Full-frame MAE/255 | Face ROI global SSIM | Face ROI MAE/255 | Изменено вне ROI |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pair_01` | 0,999696 | 0,899689 | 0,999606 | 1,839452 | 0 |
| `pair_02` | 0,998797 | 1,349565 | 0,997917 | 3,295332 | 0 |
| `pair_03` | 0,997602 | 1,292755 | 0,997305 | 3,113785 | 0 |

Минимальный full-frame SSIM — `0,997602`, минимальный face-ROI SSIM — `0,997305`;
оба выше регрессионного порога `0,95`. Android и FaceFusion визуально совпадают.
Абсолютное качество этапа C остаётся предфинальным: у `pair_02` видны переход у
лба/волос и отличие лица от шеи, у `pair_03` — мягкие переходы у волос/бороды и
различие освещения. Эти артефакты присутствуют и в эталоне. Двоение черт и смещение
глаз/рта не обнаружены; во fixtures закрыт рот, поэтому зубы не проверены. Полный
чек-лист находится в `STAGE_C_VISUAL_CHECKLIST.md`.

Артефакты этапа C:

- `reference/facefusion-3.7.1/stage_c_results.json` и `pair_*/inswapper_final_box_03.png`;
- `reference/facefusion-3.7.1/pair_*/box_mask_03.png`;
- `android/api35-x86_64/stage_c_results.json` и `pair_*_inswapper_final.png`;
- `android/api35-x86_64/pair_*_box_mask_03.png`.
