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

## Parity-ядро E1: GFPGAN 1.4 и BiSeNet на каноническом 512-кропе

### Граница проверки

Этот срез намеренно не запускает detector, alignment на Android, swapper,
композитинг, photo coordinator или UI. Он доказывает только две независимые операции
на одном byte-identical входе:

1. raw inference `gfpgan_1.4`;
2. основной output `bisenet_resnet_34` и `argmax` его 19 классов.

Геометрия уже проверена этапами B/C. Интеграция восстановления и parser-маски в
полный фото-пайплайн выполняется отдельным шагом после этого численного барьера.

### Происхождение канонического входа

Файл `inputs/pair_01_face_quality_input_512.png` получен **только десктопным
FaceFusion**, не Android-кодом:

1. `run_facefusion_reference.py` на FaceFusion 3.7.1 production-функцией
   `face_swapper.swap_face` создал уже закоммиченный Stage C кадр
   `reference/facefusion-3.7.1/pair_01/inswapper_final_box_03.png`;
2. его SHA-256 перед extraction проверяется как
   `eeada935b979fd02c34504a777681d618fd062ec3949117fa336c25d2b026afe`;
3. pinned desktop-функция
   `warp_face_by_face_landmark_5(frame, target_landmarks, "ffhq_512", (512,512))`
   использует target landmarks из `pair_01/metadata.json`;
4. PNG записывается, проверяется и **заново декодируется с диска** до inference.
   Android test читает те же закоммиченные PNG-байты.

Канонический PNG: 350 560 байт, SHA-256
`5987781f96010ceddbf7445b26bb5420b56e20138d9603a352a48a57f0fb2ec8`.
SHA-256 декодированных desktop BGR-пикселей:
`42239023bbb07502fd60efb589a5186c94b5207401be2b488976076b3ee84eea`.
Матрица desktop FaceFusion:

```text
[[ 0.5064969784282285,  0.00663733526187763, -64.4590473770804  ],
 [-0.00663733526187763, 0.5064969784282285,  -24.847903691598972]]
```

Таким образом, сам E1 parity-прогон swapper не вызывает: его вход — уже готовый,
зафиксированный desktop-кроп.

### Desktop reference

Используется тот же checkout FaceFusion 3.7.1 / commit
`3f81a8a78454089d720b8f318a12ae1702c4633b`, Python 3.12.10, ONNX Runtime 1.26.0,
OpenCV 4.13.0, NumPy 2.2.1 и `CPUExecutionProvider`. Скрипт вызывает production
`face_enhancer.prepare_crop_frame`, `face_enhancer.forward` и
`face_masker.create_region_mask`; сессии GFPGAN/BiSeNet открываются последовательно.

| Модель | Размер, байт | SHA-256 |
| --- | ---: | --- |
| `gfpgan_1.4.onnx` | 340 299 087 | `accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385` |
| `bisenet_resnet_34.onnx` | 93 632 546 | `4a0b8c958a3c938913bd06a8365dbb3c8761afba6ecbf0d14b3b1f77eb230c96` |

```powershell
$FaceFusionRoot = 'C:\Temp\facefusion-3.7.1'
$ModelDir = 'C:\Users\ozr\AppData\Local\FaceSwapLocal\models'
$ParityVenv = 'C:\Users\ozr\AppData\Local\FaceSwapLocal\facefusion-3.7.1-venv'

& "$ParityVenv\Scripts\python.exe" .\docs\parity\make_face_quality_reference.py `
  --facefusion-root $FaceFusionRoot `
  --model-dir $ModelDir `
  --parity-root .\docs\parity `
  --regenerate-canonical-input
```

Последний desktop-прогон дал:

- GFPGAN input tensor SHA-256:
  `8e41a8d62bc65d43bcc96e80fa029e056a2e32f22980e5f014960950628ff606`;
- GFPGAN raw output SHA-256:
  `90bb757e7e40533c5f46dd63b9e255a6d7452a9a7eaa34a905e888a247c86da4`;
- BiSeNet input tensor SHA-256:
  `9385bcae886b03cd59dbde9ae0215c5e49444220074c9f961f222ac8052f85c9`;
- BiSeNet argmax SHA-256:
  `a087bcca8f74103c253fb495c9242460a6a6caad88d75ce4cab8921b25c3936b`;
- CPU inference: GFPGAN 12 828 мс, BiSeNet 1 417 мс.

Полные данные находятся в
`reference/facefusion-3.7.1/face_quality/pair_01/metadata.json`.

### Android API 35 parity

Веса не входят в APK/androidTest assets. Для developer parity они помещаются через
`adb` в `files/models` debug-приложения, после чего isolated core перед **каждой**
сессией проверяет каноническое имя, размер и полный SHA-256. Product picker и UI на
этом шаге не изменяются.

```powershell
.\gradlew.bat assembleDebug assembleDebugAndroidTest --no-daemon --console=plain
adb install -r .\app\build\outputs\apk\debug\app-debug.apk
adb install -r .\app\build\outputs\apk\androidTest\debug\app-debug-androidTest.apk

adb push "$ModelDir\gfpgan_1.4.onnx" /data/local/tmp/gfpgan_1.4.onnx
adb push "$ModelDir\bisenet_resnet_34.onnx" /data/local/tmp/bisenet_resnet_34.onnx
adb shell run-as com.faceswaplocal.app mkdir -p files/models
adb shell run-as com.faceswaplocal.app cp /data/local/tmp/gfpgan_1.4.onnx files/models/gfpgan_1.4.onnx
adb shell run-as com.faceswaplocal.app cp /data/local/tmp/bisenet_resnet_34.onnx files/models/bisenet_resnet_34.onnx
adb shell rm /data/local/tmp/gfpgan_1.4.onnx /data/local/tmp/bisenet_resnet_34.onnx

adb shell am instrument -w -r `
  -e class com.faceswaplocal.app.inference.FaceQualityParityInstrumentedTest `
  com.faceswaplocal.app.test/androidx.test.runner.AndroidJUnitRunner
```

Обязательные gates: GFPGAN raw SSIM ≥ 0,95 без clamp raw tails; BiSeNet class
agreement ≥ 0,95; IoU protected-region set ≥ 0,95. Android metrics и PNG после
успешного прогона сохраняются в `android/api35-x86_64/face_quality/`.

Фактический прогон: AVD API 35 x86_64, ONNX Runtime Android 1.26.0,
`airplane_mode_on=1`, CPU с одним intra-op потоком; `OK (1 test)`, 57,476 с.

| Метрика | Результат | Gate |
| --- | ---: | ---: |
| GFPGAN raw SSIM | `0.9999999999997585` | ≥ `0.95` |
| GFPGAN raw MAE | `1.0947659726904628e-7` | диагностическая |
| GFPGAN raw max abs error | `1.6689300537109375e-6` | диагностическая |
| BiSeNet class agreement | `1.0` | ≥ `0.95` |
| BiSeNet protected-region IoU | `1.0` | ≥ `0.95` |
| GFPGAN Android inference | `47 288 ms` | диагностическая |
| BiSeNet Android inference | `6 838 ms` | диагностическая |

Все три обязательных gate пройдены. Android и desktop GFPGAN PNG визуально
неразличимы; Android BiSeNet PNG показывает raw binary protected-region set, тогда как
desktop PNG дополнительно показывает production Gaussian blur из
`create_region_mask`. Численно обе стороны сравниваются до blur по `argmax` и дают
полное совпадение.

### Android `ffhq_512` geometry spot-check

После raw parity отдельно проверена Android-геометрия без любого ONNX inference.
`FaceQualityGeometryInstrumentedTest` читает тот же desktop Stage C frame и те же пять
target landmarks, строит `FaceGeometry.estimateSimilarity(..., WarpTemplate.FFHQ_512,
512, 512)`, затем вызывает production sampler приложения
`BitmapSampling.warpAffine` (bilinear inverse sampling с edge replication).

```powershell
adb shell am instrument -w -r `
  -e class com.faceswaplocal.app.inference.FaceQualityGeometryInstrumentedTest `
  com.faceswaplocal.app.test/androidx.test.runner.AndroidJUnitRunner
```

Фактический прогон на том же AVD API 35 x86_64: `OK (1 test)`, 4,032 с.

| Метрика | Результат | Порог |
| --- | ---: | ---: |
| Максимальное расхождение проекций 5 landmarks | `0.0000027163 px` | ≤ `2 px` |
| Максимальная абсолютная ошибка affine coefficient | `0.0000070771` | диагностическая |
| SSIM Android crop ↔ canonical desktop crop | `0.9972936339` | ≥ `0.95` |
| RGB MAE | `0.448729 / 255` | диагностическая |

Порог §11.4 пройден без изменения шаблона, сэмплера или допуска. Непобитовое различие
ожидаемо: desktop использует OpenCV `INTER_AREA`, Android — собственный bilinear
sampler. Визуально кропы неразличимы. Полные matrices и hashes сохранены в
`android/api35-x86_64/face_quality/face_quality_geometry_results.json`, Android PNG —
в `android/api35-x86_64/face_quality/android_ffhq_512_crop.png`.

## E1, контрольная точка 1: two-pass coordinator и ArcFace identity

### Проверяемая конфигурация

Контрольная точка использует групповой fixture `inputs/stage_d_group_target.png`:
T1←source 1, T2←source 2, T3←source 3, T4=`Не менять`. На Android выполнены два
полных прогона одного production coordinator:

1. сила восстановления `0`: только три последовательных swap;
2. сила `0.8`: те же три swap, затем восстановление только T1/T2/T3.

FaceFusion desktop по умолчанию улучшает все найденные лица. Здесь это намеренно не
повторяется: требование приложения сильнее, и pass enhancer перечисляет только успешно
заменённые назначения. Поэтому group-frame не сравнивается с default-конфигурацией
FaceFusion как byte/SSIM golden; raw GFPGAN/BiSeNet parity и `ffhq_512` geometry уже
доказаны предыдущими независимыми барьерами.

```powershell
adb shell am instrument -w -r `
  -e class com.faceswaplocal.app.inference.StageETwoPassCoordinatorInstrumentedTest `
  com.faceswaplocal.app.test/androidx.test.runner.AndroidJUnitRunner
```

Фактический прогон: AVD API 35 x86_64, ONNX Runtime Android 1.26.0, CPU,
`airplane_mode_on=1`; `OK (1 test)`, 507,194 с. Coordinator-время: 147 820 мс без
restoration и 311 842 мс при силе 0.8.

### Lifetime и пространственные инварианты

Реальный listener записывал событие `close` только после вызова `OrtSession.close()`.
В прогоне 0.8 все три `close:inswapper_128_fp16.onnx` завершились до первого
`open:gfpgan_1.4.onnx`; затем для T1/T2/T3 шли последовательные пары GFPGAN→BiSeNet.
Максимум одновременно открытых среди InSwapper/GFPGAN/BiSeNet — `1`. При силе 0
GFPGAN и BiSeNet не проверялись и не открывались.

Чтобы nearby-незаменяемое лицо нельзя было затронуть даже при пересечении кропов,
координатор вычисляет полный потенциальный `ffhq_512` ROI каждого неназначенного
detected face и передаёт его compositor как глобальную no-write область. В этом прогоне
T4 ROI `[1062,539,1361,838)` явно присутствует в `protected_unassigned_rois`.

| Инвариант | Сила 0 | Сила 0.8 |
| --- | ---: | ---: |
| Изменено пикселей вне `union(swap ROI, enhance ROI)` | 0 | 0 |
| Изменено пикселей в полном потенциальном `ffhq_512` ROI T4 | 0 | 0 |
| Enhance target IDs | — | T1, T2, T3 |

### ArcFace identity

Для сравнения использованы исходные пять landmarks целевого кадра; финальные лица не
редетектировались, поэтому detector drift не влияет на результат. Значения — cosine
между ArcFace target embedding и тремя source embeddings:

| Target / сила | Source 1 | Source 2 | Source 3 | Ближайший | Margin ожидаемого source |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 / `0` | 0.813303 | 0.177419 | 0.292314 | S1 | 0.520990 |
| T1 / `0.8` | 0.783231 | 0.196483 | 0.297674 | S1 | 0.485556 |
| T2 / `0` | 0.163611 | 0.886655 | 0.074729 | S2 | 0.723044 |
| T2 / `0.8` | 0.182493 | 0.847742 | 0.093036 | S2 | 0.665249 |
| T3 / `0` | 0.265036 | 0.056614 | 0.865945 | S3 | 0.600909 |
| T3 / `0.8` | 0.280638 | 0.086452 | 0.807063 | S3 | 0.526425 |

GFPGAN снизил абсолютное сходство с ожидаемым source на `0.030073`, `0.038913` и
`0.058883`, но ни для одной цели не изменил ближайший source или полный порядок
ранжирования. Условие снижения дефолтной силы не сработало: `0.8` остаётся допустимым
кандидатом для E2.

Артефакты:

- `android/api35-x86_64/checkpoint_1/checkpoint_1_results.json` — полные ROI, матрицы
  сходства, delta и open/close events;
- `android/api35-x86_64/checkpoint_1/strength_0.png` — baseline;
- `android/api35-x86_64/checkpoint_1/strength_0_8.png` — результат восстановления.
