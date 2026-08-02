# Отчёт: Этап E1 — Качество лица
Дата: 02.08.2026, ветка/коммит: `main` / `9ac1645fa63b0f97f2cbebd5e51b239214f2ec39`

## 1. Сделано

- Подключены локальные `gfpgan_1.4` и `bisenet_resnet_34` через официальный ONNX
  Runtime Android. Веса остаются вне Git/APK и открываются только после проверки
  приватной копии в `ModelStore`.
- Создано raw parity-ядро на одном каноническом desktop-кропе `ffhq_512`; GFPGAN и
  BiSeNet на Android численно сравнены с FaceFusion 3.7.1 без участия swapper, UI и
  координатора.
- Android-геометрия `WarpTemplate.FFHQ_512` отдельно проверена на том же кадре этапа C:
  production affine и bilinear sampler проходят порог §11.4 без подгонки шаблона или
  допуска.
- Фото-координатор переведён на два прохода: все InSwapper-композиты выполняются
  первыми; InSwapper закрывается до открытия GFPGAN; восстановление перечисляет только
  назначенные цели. T4 с состоянием «Не менять» остаётся побитово идентичен.
- Одна BiSeNet-сессия переиспользуется всей задачей. Допустимые пиковые пары —
  BiSeNet+InSwapper в первом проходе и BiSeNet+GFPGAN во втором; InSwapper и GFPGAN
  одновременно не существуют.
- Сила восстановления проверена при `0`, `0.8` и `1.0`. Для T1/T2/T3 ожидаемый
  источник остался ближайшим по ArcFace при всех трёх значениях; default `0.8`
  сохранён.
- Parser-mask подключена к alpha самого свапа. На `pair_02` и `pair_03` она убрала
  переходы box-mask у волос, виска и нижней границы бороды, не добавив двоения или
  смещения глаз/рта.
- В UI добавлены включение GFPGAN, slider силы `0…1` с default `0.8` и переключатель
  parser/box-маски свапа. Настройки сохраняются через `SavedStateHandle`, блокируются
  во время обработки и преобразуются в effective strength, набор обязательных моделей
  и `SwapBlendMaskMode` production-вызова.
- Добавлено сравнение «До/После» в одном viewport без автосохранения. При повторной
  обработке старый результат остаётся доступен до атомарной публикации нового, после
  выхода из Compose его bitmap освобождается.
- Закрыта граница поздней отмены: если coordinator уже вернул результат, но coroutine
  отменена до публикации, непоказанный bitmap освобождается в `finally`; unit-тесты
  подтверждают один release при отмене и отсутствие release после публикации.
- Снята диагностика цвета без изменения production-алгоритма. Подтверждено, что
  RGB mean/std считается по box-mask, а parser покрывает около 54% её веса. Разница
  face↔neck измерена после swap+color и после GFPGAN; коэффициенты по синтетическим
  fixtures не менялись.
- Definition of Done E1 — **ВЫПОЛНЕНО на утверждённом синтетическом parity-наборе**:
  GFPGAN output численно и визуально совпадает с desktop reference, parser убирает
  видимую геометрическую границу на контрольных `pair_02`/`pair_03`, а ArcFace
  подтверждает сохранение source identity при рабочей и максимальной силе.
- Полный FR-PHOTO-05 не объявляется закрытым: presets, identity strength, ручной
  feathering, color auto/off и watermark остаются в E2. Сравнение «До/После» выполнено
  досрочно по прямому указанию пользователя.

## 2. Проверки

| Проверка | Команда | Результат |
| --- | --- | --- |
| Unit tests | `.\gradlew.bat test --no-daemon --console=plain` | OK — 56 уникальных тестов, 112 debug/release прогонов, 0 failures/errors/skipped |
| Lint | `.\gradlew.bat lint --no-daemon --console=plain` | OK — 0 errors, 32 warnings |
| Сборка | `.\gradlew.bat assembleDebug --no-daemon --console=plain` | OK — `app-debug.apk`, 180 617 815 байт, SHA-256 `C68366F1F3D8299A792D5327EA2DB54E6B13F42FA84AB8A0060A453D5E418347` |
| Raw GFPGAN/BiSeNet parity | `adb shell am instrument ... -e class com.faceswaplocal.app.inference.FaceQualityParityInstrumentedTest` | OK — 1 test, 57,476 с |
| Android `ffhq_512` geometry | `adb shell am instrument ... -e class com.faceswaplocal.app.inference.FaceQualityGeometryInstrumentedTest` | OK — 1 test, 4,032 с |
| Two-pass + identity | `adb shell am instrument ... -e class com.faceswaplocal.app.inference.StageETwoPassCoordinatorInstrumentedTest` | OK — 1 test, 887,171 с |
| Parser-mask swap | `adb shell am instrument ... -e class com.faceswaplocal.app.inference.StageESwapParserMaskInstrumentedTest` | OK — 2 tests, 414,922 с |
| Диагностика цвета | `adb shell am instrument ... -e class com.faceswaplocal.app.inference.StageEColorDiagnosticsInstrumentedTest` | OK — 1 test, 321,736 с |
| UI качества | `adb shell am instrument ... -e class com.faceswaplocal.app.ui.StageEFaceQualityUiInstrumentedTest` | OK — 3 tests, финальный прогон 208,019 с |
| Manifest/APK | merged manifest + `jar tf app-debug.apk` | OK — `INTERNET`/`ACCESS_NETWORK_STATE`: 0; `.onnx` в APK: 0 |
| Устройство | `adb install -r app-debug.apk` + запуск/ручной UI smoke | OK — API 35 x86_64, `airplane_mode_on=1`, `MainActivity` в `topResumedActivity`; настройки и оба состояния «До/После» отображаются |

Финальная обязательная команда после последней кодовой правки:

```powershell
.\gradlew.bat test lint assembleDebug --no-daemon --console=plain
```

Результат: `BUILD SUCCESSFUL` за 7 мин 53 с. Из-за параллельного точечного аудита
debug XML был перезаписан фильтрованным прогоном; после завершения аудита отдельно
выполнен `testDebugUnitTest --rerun-tasks` — `BUILD SUCCESSFUL`, все 56 debug-тестов.
Release XML содержит те же 56 тестов; суммарно 112 variant executions.

Скриншоты API 35 на синтетическом debug harness:

- `docs/reports/img/STAGE_E1_SETTINGS_API35.png`;
- `docs/reports/img/STAGE_E1_BEFORE_AFTER_API35.png`;
- `docs/reports/img/STAGE_E1_COMPARE_BEFORE_API35.png`.

## 3. Изменённые файлы

- `app/src/main/java/com/faceswaplocal/app/inference/OnnxFaceEnhancerPipeline.kt` —
  GFPGAN preprocessing, inference, postprocessing и смешивание с заданной силой.
- `app/src/main/java/com/faceswaplocal/app/inference/OnnxFaceParserPipeline.kt` —
  BiSeNet preprocessing, argmax классов и region-mask.
- `app/src/main/java/com/faceswaplocal/app/inference/OnnxFaceQualityParityCore.kt` —
  изолированное raw parity-ядро E1.
- `app/src/main/java/com/faceswaplocal/app/inference/OnnxMultiPhotoFaceSwapPipeline.kt` —
  двухпроходный coordinator, session lifetime, восстановление только назначений и
  защита no-write ROI.
- `app/src/main/java/com/faceswaplocal/app/inference/FaceCompositor.kt` — parser-guided
  alpha свапа и восстановления; production color matching не менялся в checkpoint 3.
- `app/src/main/java/com/faceswaplocal/app/inference/FaceGeometry.kt` — шаблон
  `FFHQ_512` и геометрия enhancer crop.
- `app/src/main/java/com/faceswaplocal/app/inference/InferenceSessionLifecycle.kt` —
  проверяемый журнал реального открытия/закрытия тяжёлых сессий.
- `app/src/main/java/com/faceswaplocal/app/inference/ModelCatalog.kt`, `ModelStore.kt` —
  каталожные записи GFPGAN/BiSeNet и безопасное открытие локальных моделей.
- `app/src/main/java/com/faceswaplocal/app/domain/FaceProcessing.kt` — immutable
  `FaceQualitySettings` и безопасные defaults.
- `app/src/main/java/com/faceswaplocal/app/ui/FaceSwapViewModel.kt` — сохранение и
  передача настроек, динамический набор моделей, result ownership и cleanup отмены.
- `app/src/main/java/com/faceswaplocal/app/ui/FaceSwapApp.kt` — настройки качества и
  общий viewport «До/После».
- `app/src/debug/java/com/faceswaplocal/app/ui/StageDUiTestActivity.kt` — полностью
  синтетический UI harness без picker и inference.
- `app/src/test/java/com/faceswaplocal/app/inference/FaceRegionMaskTest.kt`,
  `FaceCompositorTest.kt` — mask/compositing unit-регрессии.
- `app/src/test/java/com/faceswaplocal/app/ui/FaceQualitySettingsTest.kt` — defaults,
  persistence, model/mask mapping и ownership при поздней отмене.
- `app/src/androidTest/java/com/faceswaplocal/app/inference/FaceQuality*`,
  `StageETwoPassCoordinatorInstrumentedTest.kt`,
  `StageESwapParserMaskInstrumentedTest.kt`,
  `StageEColorDiagnosticsInstrumentedTest.kt` — geometry/raw/production/color proofs.
- `app/src/androidTest/java/com/faceswaplocal/app/ui/StageEFaceQualityUiInstrumentedTest.kt`
  — settings, recreate, running lock, before/after и bitmap lifetime.
- `docs/parity/inputs/pair_01_face_quality_input_512.png`,
  `docs/parity/make_face_quality_reference.py`,
  `docs/parity/reference/facefusion-3.7.1/face_quality/` — canonical desktop input,
  воспроизводимый reference и raw tensors.
- `docs/parity/android/api35-x86_64/face_quality/`, `checkpoint_1/`, `checkpoint_2/`,
  `checkpoint_3/` — Android metrics, PNG и session events четырёх E1 проверок.
- `docs/parity/README.md`, `STAGE_C_VISUAL_CHECKLIST.md`, `docs/BENCHMARKS.md`,
  `docs/MODEL_CARD.md`, `docs/KNOWN_LIMITATIONS.md`, `README.md` — методика, результаты,
  ограничения и актуальное состояние продукта.
- `THIRD_PARTY_NOTICES.md` — источники и лицензии E1-моделей/reference.
- `docs/reports/STAGE_E1_REPORT.md` — настоящий отчёт.

## 4. Benchmark и parity

- Canonical desktop input: `pair_01_face_quality_input_512.png`, SHA-256
  `5987781F96010CEDDBF7445B26BB5420B56E20138D9603A352A48A57F0FB2EC8`;
  он получен FaceFusion 3.7.1 из того же кадра этапа C.
- Raw GFPGAN: SSIM `0,9999999999997585`, MAE `1,0948e-7`; Android/desktop PNG
  визуально неразличимы. BiSeNet: class agreement `1,0`, protected-region IoU `1,0`.
- Android `ffhq_512`: максимальная ошибка проекций landmarks `0,0000027163 px`,
  SSIM кропа к desktop `0,9972936339`, RGB MAE `0,448729/255`.
- ArcFace margin ожидаемого source к ближайшему неправильному source:

| Цель | Сила `0` | Сила `0.8` | Сила `1.0` |
| --- | ---: | ---: | ---: |
| T1→S1 | 0,520990 | 0,485556 | 0,465453 |
| T2→S2 | 0,723044 | 0,665249 | 0,645125 |
| T3→S3 | 0,600909 | 0,526425 | 0,502646 |

- Parser proof: MAE к target в исключённой parser-полосе уменьшился с `11,930633`
  до `0,013237` на `pair_02` и с `11,039550` до `0,007989` на `pair_03`; вне paste
  ROI изменено 0 пикселей.
- Пиковые пары файлов весов: BiSeNet+InSwapper — 371 313 375 байт;
  BiSeNet+GFPGAN — 433 931 633 байт. Это не peak heap; InSwapper+GFPGAN одновременно
  не открывались.
- Color statistics использует box-mask: parser overlap веса — 53,952865% (`pair_02`)
  и 53,880621% (`pair_03`). Тон face↔neck:

| Пара | Target ΔE00 | После swap+color ΔE00 | После GFPGAN 0.8 ΔE00 | Residual swap, ΔE76 | Residual GFPGAN, ΔE76 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pair_02` | 7,304500 | 8,681741 | 7,908405 | 3,949355 | 2,503508 |
| `pair_03` | 31,162696 | 28,345982 | 27,434374 | 3,992223 | 6,033606 |

- На `pair_02` GFPGAN улучшил отношение face↔neck, на `pair_03` ухудшил; neck ROI
  после обоих проходов изменил 0 пикселей. Это смешанный сигнал, поэтому luminance-only
  restoration и новые коэффициенты не внедрялись.
- Устройство измерений: AVD `QuizMaster_API35`, Android 15/API 35, x86_64, 2 vCPU,
  около 2 ГБ RAM, ONNX Runtime 1.26.0 CPU, авиарежим. Peak Java/native heap и thermal
  не измерялись; AVD не является reference device.

## 5. Отклонения от ТЗ

- По прямому указанию пользователя настройки силы/маски и сравнение «До/После»
  перенесены из E2 в контрольную точку 3 и выполнены досрочно. Остальные задачи E2 не
  объявляются готовыми.
- Desktop FaceFusion по умолчанию может восстанавливать все обнаруженные лица;
  Android намеренно восстанавливает только назначенные цели, чтобы «Не менять»
  оставалось побитовым инвариантом.
- Android color matching является требованием FR-PHOTO-06, но отсутствует в
  использованном FaceFusion Stage C reference. Его статистика остаётся на box-mask;
  checkpoint 3 только измерил последствия, как потребовал пользователь.
- Реализована BiSeNet face-region parser-mask. Отдельная occlusion/xseg-модель для рук,
  сложных перекрытий и прозрачных оправ не добавлялась без benchmark на physical
  reference device.
- Все E1 quality/parity проверки используют синтетические лицензированно чистые
  fixtures. Коэффициенты по их плоскому тону шеи не подгонялись.

## 6. Известные проблемы и ограничения

- Box-mask color statistics примерно на 46% веса находится вне parser-region. На
  реальном фото нужно отдельно сравнить parser-scoped statistics и сохранение цветности
  при GFPGAN; текущие две синтетические пары дают противоположный результат.
- InSwapper формирует raw crop 128×128. GFPGAN устраняет характерную мягкость на
  parity-наборе, но резкость и identity требуют повторной проверки на разрешённом
  реальном фото первого physical reference device.
- Нет отдельной occlusion/xseg-модели и fixture с видимыми зубами; сложные руки,
  перекрытия, прозрачные очки и открытый рот полностью не закрыты.
- Peak Java/native heap, 12 МП без OOM, пять повторных циклов и thermal throttling не
  проверены без подключённого ARM reference device; по TECHNICAL_SPEC.md 2.1 это
  отложенные пункты E2, а не выполненные проверки E1.
- На перегруженном 2-ГБ AVD после длительных inference-прогонов один раз завис System
  UI; после выбора `Wait` эмулятор восстановился. Финальный `am start -W` также вернул
  timeout, но `dumpsys` подтвердил `MainActivity` в `topResumedActivity`; падения
  процесса приложения не было.
- Presets «Быстро/Баланс/Максимум», identity strength, feathering, color auto/off,
  watermark, progress/cancel UI, MediaStore export и полный cleanup/profile остаются E2.

## 7. Блокеры

- Нет. Отсутствие physical ARM reference device не блокирует E1 по текущей разбивке
  TECHNICAL_SPEC.md 2.1; перечисленные device-only проверки явно отложены до E2.

## 8. Следующий шаг

- После подтверждения пользователя выполнить этап E2: реальные presets качества,
  progress/cancel с cleanup, MediaStore export и профилирование; device-only пункты
  закрыть после подключения первого физического ARM reference device.
