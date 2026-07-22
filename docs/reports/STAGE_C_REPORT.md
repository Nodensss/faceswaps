# Отчёт: Этап C — Первый настоящий свап на фото
Дата: 22.07.2026, ветка/коммит: `main` / `487ba428e654e4b86f3cf76021bd8e28e945da99`

## 1. Сделано

- Реализован `OnnxPhotoFaceSwapPipeline`: проверенный пайплайн этапа B запускает
  `inswapper_128_fp16`, после чего сырой crop возвращается в координаты целевого фото
  через inverse affine transform. Результат сохраняет исходные размеры цели.
- Реализован `FaceCompositor`: FaceFusion-compatible affine box-mask с blur `0.3`,
  Gaussian feathering и `REFLECT_101`, inverse warp маски с нулевой границей, inverse
  warp crop с edge replicate и мягкий alpha blend. Пиксели вне вычисленного paste ROI
  остаются побитово неизменными.
- Реализовано обязательное цветосогласование: masked RGB mean/std, ограниченный gain
  `0.85…1.15`, offset `±24` и сила `0.65`. Альфа целевого bitmap сохраняется.
- Связь UI assignment с нейродетектором сделана явной: выбранные нормализованные рамки
  ML Kit передаются только как spatial hint, а YOLOFace по-прежнему самостоятельно
  получает 5 landmarks. Среди YOLO-кандидатов выбирается максимальный IoU с выбранной
  рамкой; не пересекающееся лицо отклоняется вместо молчаливой подмены другого человека.
- Compose/ViewModel переведены с показа raw crop на полный сценарий «одно лицо-источник
  → одно целевое лицо»: системный Photo Picker → ML Kit preview → assignment →
  локальный InSwapper → полноразмерный вставленный результат. UI показывает размер,
  backend и времена detector/embedding/swapper/blend и явно сообщает, что результат
  ещё не сохранён; экспорт остаётся границей этапа E.
- CPU fallback обязателен в пользовательском пути. Запрашивается
  `XNNPACK_WITH_CPU_FALLBACK`; известные небезопасные x86/x86_64 ABI переводятся в
  `CPU_FALLBACK` до создания native session, а перехватываемая `OrtException` на других
  ABI повторяет операцию на CPU.
- Bitmap/session lifetime закрыт: тяжёлые ONNX-сессии последовательно открываются и
  закрываются в `use`; raw/aligned bitmap освобождаются после compositing; предыдущий
  UI-результат освобождается при повторном запуске, смене media и `onCleared`.
  Дополнительно устранён prompt-cancellation race `withContext`: готовый, но не
  доставленный вызывающей стороне полноразмерный bitmap освобождается outer `finally`.
- Desktop runner расширен production-вызовами FaceFusion 3.7.1
  `face_swapper.swap_face`, `create_box_mask` и `paste_back`. Для трёх синтетических
  пар сохранены canonical final frames, box masks, конфигурация и SHA-256; веса
  моделей остались вне Git/APK.
- Добавлен Android final-frame parity test для трёх пар. Он проверяет размеры,
  обязательный CPU backend, диапазон/центр/границу маски, ноль изменений вне inverse
  ROI, фактическое изменение лица, full-frame SSIM и face-ROI SSIM не ниже `0.95`.
- Добавлены 10 unit-тестов compositing: box mask, symmetry, Gaussian boundary,
  bounded color match, alpha, identity/translated/fractional affine, ROI, неизменность
  входов и invalid inputs. Добавлены 4 теста выбора YOLO-лица по UI hint и IoU.
- В `docs/parity/STAGE_C_VISUAL_CHECKLIST.md` отдельно зафиксированы mobile↔desktop
  parity и абсолютное предфинальное качество для границы, тона кожи, двоения черт,
  глаз, рта/зубов, identity и позы.
- Definition of Done — **ВЫПОЛНЕНО:** результат действительно меняет identity выбранного
  целевого лица через локальный InSwapper, а не копирует crop и не рисует заглушку.
- Definition of Done — **ВЫПОЛНЕНО:** поза, направление взгляда и выражение цели
  сохраняются на parity-наборе; identity источника визуально узнаваема во всех трёх
  парах.
- Definition of Done — **ВЫПОЛНЕНО:** полный пользовательский сценарий воспроизведён
  на API 35 x86_64 в авиарежиме; backend в UI — `CPU_FALLBACK` для detector, ArcFace
  и InSwapper.
- Definition of Done — **ВЫПОЛНЕНО:** final-frame Android↔FaceFusion parity пройдена;
  минимальный face-ROI SSIM `0.997305`, вне paste ROI изменено `0` пикселей.
- Definition of Done — **ВЫПОЛНЕНО в границах этапа C:** результат узнаваемый, но
  резкость 128×128, границы волос/бороды и restoration остаются предфинальными и
  честно перенесены в ограничения этапа E.

## 2. Проверки

| Проверка | Команда | Результат |
| --- | --- | --- |
| Unit tests | `.\gradlew.bat test --console=plain` | OK — 38 тестов в debug и 38 в release, 76 успешных прогонов, 0 failures/errors/skipped |
| Lint | `.\gradlew.bat lint --console=plain` | OK — 0 errors, 32 warnings: 27 `GradleDependency`, 3 `AndroidGradlePluginVersion`, 1 `DataExtractionRules`, 1 `OldTargetApi`; обновления отложены до этапа G |
| Сборка | `.\gradlew.bat assembleDebug --console=plain` | OK — `app-debug.apk`, 179 776 628 байт; SHA-256 `2798a29b36ca6fa3dde9c57bd88c436fb99842e49877170769bedeced0cc683f` |
| Final-frame parity | `adb -s emulator-5554 shell am instrument -w -r -e class com.faceswaplocal.app.inference.FaceFusionFinalFrameParityInstrumentedTest com.faceswaplocal.app.test/androidx.test.runner.AndroidJUnitRunner` | OK — финальный код и APK, 3 полных InSwapper CPU-прогона; `Time: 224.507`, `OK (1 test)` |
| Устройство | `adb -s emulator-5554 install -r app/build/outputs/apk/debug/app-debug.apk` + ручной сценарий | OK — API 35 x86_64: Photo Picker → лица найдены → `Целевое лицо 1 → Источник 1` → полноразмерный swap 1254×1254; `airplane_mode_on=1`; финальный APK повторно установлен и запущен |
| Приватность и упаковка | проверка merged manifest, `git ls-files`, список APK через `jar tf` | OK — `INTERNET` и `ACCESS_NETWORK_STATE` отсутствуют; ONNX-весов в Git и APK нет |
| Desktop reference | запуск `docs/parity/run_facefusion_reference.py` в pinned FaceFusion 3.7.1 venv | OK — 3 canonical final frames, masks, JSON/metadata; `py_compile` и проверка 7 JSON-файлов успешны |

## 3. Изменённые файлы

- `app/src/main/java/com/faceswaplocal/app/inference/FaceCompositor.kt` — affine mask,
  Gaussian feathering, color matching, inverse warp и alpha blending.
- `app/src/main/java/com/faceswaplocal/app/inference/OnnxPhotoFaceSwapPipeline.kt` —
  полный Stage C pipeline, InSwapper, CPU fallback, timings и bitmap ownership.
- `app/src/main/java/com/faceswaplocal/app/inference/OnnxRawFaceSwapPipeline.kt` —
  source/target face hints, выбор YOLO-кандидата по IoU и защита assignment.
- `app/src/main/java/com/faceswaplocal/app/ui/FaceSwapViewModel.kt` — coroutine orchestration,
  one-to-one gate, CPU fallback и lifetime полноразмерного результата.
- `app/src/main/java/com/faceswaplocal/app/ui/FaceSwapApp.kt` — Stage C UI запуска,
  полноразмерный preview и backend/timing diagnostics.
- `app/src/main/java/com/faceswaplocal/app/domain/FaceProcessing.kt` — актуализирован
  комментарий реального inference-контракта.
- `app/src/test/java/com/faceswaplocal/app/inference/FaceCompositorTest.kt` — 10 тестов
  compositing, mask, color и affine boundaries.
- `app/src/test/java/com/faceswaplocal/app/inference/FaceSelectionTest.kt` — 4 теста
  выбора назначенного лица и IoU.
- `app/src/androidTest/java/com/faceswaplocal/app/inference/FaceFusionFinalFrameParityInstrumentedTest.kt`
  — Android final-frame parity для трёх пар.
- `docs/parity/run_facefusion_reference.py` — production final-frame reference;
  `docs/parity/reference/facefusion-3.7.1/desktop_results.json`,
  `pair_*/metadata.json` и `stage_c_results.json` — конфигурация и метрики эталона.
- `docs/parity/reference/facefusion-3.7.1/pair_*/inswapper_final_box_03.png` и
  `box_mask_03.png` — три эталонных финальных кадра и маски.
- `docs/parity/android/api35-x86_64/pair_*_inswapper_final.png`,
  `pair_*_box_mask_03.png` и `stage_c_results.json` — Android outputs и метрики.
- `docs/parity/README.md`, `docs/parity/STAGE_C_VISUAL_CHECKLIST.md` — команды,
  численная parity, documented deviation и визуальная проверка артефактов.
- `docs/BENCHMARKS.md`, `docs/KNOWN_LIMITATIONS.md`, `README.md` — состояние,
  измерения и ограничения после этапа C.
- `docs/reports/img/STAGE_C_FINAL_API35.png`,
  `docs/reports/img/STAGE_C_FINAL_DETAILS_API35.png` — ручной UI-сценарий на синтетике.
- `docs/reports/STAGE_C_REPORT.md` — отчёт этапа C.

## 4. Benchmark и parity (если предусмотрены этапом)

- Desktop reference: FaceFusion 3.7.1, commit
  `3f81a8a78454089d720b8f318a12ae1702c4633b`, Python 3.12.10, ONNX Runtime 1.26.0
  CPU, OpenCV 4.13.0, NumPy 2.2.1. Конфигурация: InSwapper 128 fp16, pixel boost
  `128x128`, weight `0.5`, box mask, blur `0.3`, padding `0`.
- Android: AVD API 35 x86_64, ONNX Runtime Android 1.26.0, CPU. Committed metrics:

| Пара | Full-frame SSIM | Face ROI SSIM | Face ROI MAE/255 | Изменено вне ROI | Detector, ms | ArcFace, ms | InSwapper, ms | Blend, ms | Total, ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pair_01` | 0.999696 | 0.999606 | 1.839452 | 0 | 2 918 | 3 969 | 38 752 | 1 907 | 47 760 |
| `pair_02` | 0.998797 | 0.997917 | 3.295332 | 0 | 2 391 | 6 298 | 39 847 | 942 | 49 542 |
| `pair_03` | 0.997602 | 0.997305 | 3.113785 | 0 | 2 338 | 4 178 | 39 493 | 923 | 46 978 |

- Минимальный full-frame/face-ROI SSIM: `0.997602` / `0.997305`; оба выше порога
  `0.95`. Максимальный face-ROI MAE: `3.295332/255`. Вне inverse ROI изменено ровно
  `0` пикселей во всех парах.
- Финальный instrumentation повторён после последнего lifecycle-исправления:
  `Time: 224.507`, `OK (1 test)`. Значения изображения совпали; AVD timings ожидаемо
  меняются от нагрузки.
- Desktop FaceFusion `swap_face` CPU: `4 780.079`, `5 689.506`, `5 645.149` ms.
  Эти времена не сравниваются с AVD как оценка производительности телефона.
- Ручной холодный UI-прогон `pair_01` в авиарежиме: detector `16 406 ms`, ArcFace
  `23 593 ms`, InSwapper `211 152 ms`, blend `9 229 ms`, total `262 232 ms`;
  detector/ArcFace/swapper показали `CPU_FALLBACK`.
- Визуально Android неотличим от соответствующих FaceFusion final frames. `pair_01`
  имеет хороший предфинальный blend. У `pair_02` на обеих сторонах заметны переход у
  волос/виска и отличие лица от шеи; у `pair_03` — мягкость у волос/бороды и отличие
  освещения. Двоение черт и смещение глаз/рта не обнаружены. Во всех fixtures рот
  закрыт, поэтому зубы объективно не проверены.
- Физическое Android-устройство не подключено. Модель телефона, SoC, RAM, peak
  Java/native heap, thermal и ARM64/XNNPACK должны быть добавлены при появлении первого
  reference device; AVD не является прогнозом его производительности.

## 5. Отклонения от ТЗ

- Canonical FaceFusion 3.7.1 `face_swapper.swap_face` не выполняет отдельное color
  matching. Android намеренно добавляет masked RGB mean/std со strength `0.65`, потому
  что это прямое требование FR-PHOTO-06. Поэтому финальные PNG не побитово идентичны;
  измеренный минимум face-ROI SSIM `0.997305` и визуальная сверка подтверждают parity.
- Сохраняются документированные отклонения этапа B: Android использует
  детерминированный closed-form least-squares similarity transform и bilinear sampler
  вместо OpenCV RANSAC/`INTER_AREA`. Они численно покрыты geometry/raw/final parity.
- Parity использует исходные 5 landmarks YOLOFace на обеих сторонах без отдельного
  68→5 refinement. 68/106 landmarker и parsing/occlusion flow относятся к улучшенной
  маске этапа E и потребуют собственной лицензии/MODEL_CARD/parity.
- XNNPACK не запускается на x86/x86_64 AVD из-за ранее подтверждённого native
  `SIGABRT` ONNX Runtime 1.26.0; эти ABI превентивно используют CPU fallback.
  Корректность XNNPACK на ARM64 не заявляется без физического reference device.
- HyperSwap в соответствии с прямым указанием пользователя в этапе C не изменялся и
  не использовался; рабочий pipeline и final parity выполнены на InSwapper.

## 6. Известные проблемы и ограничения

- Полный UI-пайплайн пока обрабатывает только одну source/target пару; независимые
  множественные назначения относятся к этапу D.
- InSwapper crop 128×128 даёт ожидаемую предфинальную мягкость. GFPGAN 1.4,
  настраиваемая сила restoration и улучшенная face parsing/occlusion mask относятся к
  этапу E.
- Affine box-mask не защищает волосы, очки, руки и сложные перекрытия. Переходы
  `pair_02`/`pair_03` зафиксированы в визуальном чек-листе, а не скрыты как успешное
  абсолютное качество.
- Текущий parity-набор не содержит видимых зубов и сложной окклюзии; это ограничение
  покрытия golden/quality tests.
- Сравнение до/после, пользовательские настройки качества, прогресс с отменой и
  экспорт через MediaStore ещё не реализованы; они входят в этап E.
- XNNPACK/ARM64, NNAPI, peak heap, thermal throttling и пять повторных memory cycles
  не проверены без физического reference device.
- Debug APK занимает `171.45 MiB` из-за нативных ABI-библиотек ONNX Runtime; весов
  моделей в APK нет. ABI splits/minification относятся к этапу G.
- 32 lint warning относятся к версиям Gradle/AGP и Android backup/target API metadata;
  ошибок lint нет, обновления согласованно отложены до этапа G.

## 7. Блокеры

- Для CPU Definition of Done этапа C блокеров нет. Benchmark физического reference
  device и ARM64/XNNPACK ожидают подключённый телефон и не заявлены как выполненные.

## 8. Следующий шаг

- После подтверждения пользователя перейти к этапу D: пакет source-фотографий,
  «Не менять», «Применить ко всем», стабильные множественные назначения, обработка
  пересекающихся масок и сохранение UI state при rotation.
