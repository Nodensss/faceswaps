# Отчёт: Этап B — Модели подключены и проверены
Дата: 21.07.2026, ветка/коммит: `main` / `a64768e55da4bd00bea0ce13e8d1d686256e2659`

## 1. Сделано

- Shortlist §11.2 зафиксирован в `docs/MODEL_CARD.md`: `hyperswap_1a_256` как основной кандидат, `inswapper_128_fp16` как рабочий fallback, `arcface_w600k_r50` для identity embedding и `yoloface_8n` для детекции 5 точек. Источники, pinned revisions, размеры, SHA-256, tensor contracts, нормализация и ограничения лицензий документированы.
- `THIRD_PARTY_NOTICES.md` содержит дословный MIT-текст ONNX Runtime, дословное содержимое license-файла FaceFusion 3.7.1 и дословные license metadata моделей. Неоднозначные либо non-commercial условия отмечены консервативно: веса не распространяются в Git или APK.
- Реализован локальный импорт каждого ONNX-файла через системный picker: проверяются точный размер и полный SHA-256 до атомарного сохранения в приватное хранилище и повторно перед открытием session. Ошибка импорта не уничтожает ранее установленную валидную модель; `.part` очищается.
- Подключён официальный `onnxruntime-android:1.26.0`. Реализованы последовательно открываемые и закрываемые сессии YOLOFace, ArcFace и выбранного swapper; runtime-типы не попали в ViewModel. CPU — проверенный путь. Запрос XNNPACK на x86/x86_64 заранее переводится в `CPU_FALLBACK`, потому что ORT 1.26.0 аварийно завершает процесс AVD внутри native `OrtSession_createSession`; ARM64 оставлен для отдельной проверки на реальном устройстве.
- Реализован сырой пайплайн: нормализованный Android `ImageDecoder` bitmap → YOLOFace 5 landmarks → FaceFusion warp template (`arcface_112_v2` для ArcFace, `arcface_128` в масштабе 128×128 или 256×256 для swapper) → ArcFace embedding → HyperSwap либо InSwapper. Для InSwapper реализована эталонная конвертация `raw_embedding × emap` с нормализацией.
- В Compose UI добавлены статусы и picker импорта моделей, выбор swapper, запуск вне Main thread, диагностические backend/timing и показ отдельного квадратного raw crop. Жизненный цикл исходных, целевых и результирующих bitmap закрыт при замене, ошибке, отмене и `onCleared`.
- Десктопный эталон закреплён на FaceFusion 3.7.1, commit `3f81a8a78454089d720b8f318a12ae1702c4633b`. `docs/parity/run_facefusion_reference.py` вызывает production-функции FaceFusion с локальными checksum-verified ORT sessions. Parity-набор состоит из трёх полностью синтетических пар; сохранены входы, выровненные кропы, float32 tensors, raw outputs, команды и результаты.
- Добавлены unit-тесты similarity/inverse transform, warp templates, каталога и целостности моделей, сохранения валидной установленной копии при неудачной замене, чтения `emap` из ONNX protobuf и backend policy. Instrumentation parity-тест выполняет 3 пары × 2 swapper.
- Definition of Done — **ВЫПОЛНЕНО:** геометрия и raw output мобильного CPU-пайплайна численно совпадают с FaceFusion; минимальный raw SSIM = `0.997513`, что выше требуемых `0.95`.
- Definition of Done — **ВЫПОЛНЕНО для утверждённого fallback InSwapper:** во всех трёх raw crop визуально узнаётся identity источника; полный UI-сценарий повторён в авиарежиме.
- Definition of Done — **НЕ ПОДТВЕРЖДЕНО для HyperSwap:** численная parity с FaceFusion выполнена, но перенос identity на текущих синтетических fixtures визуально слабее InSwapper. Поэтому HyperSwap сохранён как основной кандидат shortlist, а рабочим UI-default выбран InSwapper.
- Definition of Done — **ВЫПОЛНЕНО:** результат этапа является отдельным квадратным crop без inverse transform, маски и вставки в целевую фотографию, как требует граница этапа B.

## 2. Проверки

| Проверка | Команда | Результат |
| --- | --- | --- |
| Unit tests | `.\gradlew.bat test --console=plain` | OK — 24 теста в debug и 24 в release, 48 успешных прогонов, 0 failures/errors/skipped |
| Lint | `.\gradlew.bat lint --console=plain` | OK — 0 errors, 32 warnings: 27 `GradleDependency`, 3 `AndroidGradlePluginVersion`, 1 `DataExtractionRules`, 1 `OldTargetApi`; обновления отложены до этапа G |
| Сборка | `.\gradlew.bat assembleDebug --console=plain` | OK — `app-debug.apk`, 179 781 996 байт; SHA-256 `3de706858569c3bf161df038371af28a5f9dc531c72ab507a1c6d65ac00a3442` |
| Parity instrumentation | `adb -s emulator-5554 shell am instrument -w -r -e class com.faceswaplocal.app.inference.FaceFusionParityInstrumentedTest com.faceswaplocal.app.test/androidx.test.runner.AndroidJUnitRunner` | OK — 1 instrumentation method, 6 полных inference-прогонов; `Time: 201.851`, `OK (1 test)` |
| Устройство | `adb install -r app/build/outputs/apk/debug/app-debug.apk` + ручной сценарий на API 35 | OK — системный Photo Picker → по одному лицу найдено → `Целевое лицо 1 → Источник 1` → raw InSwapper crop; `airplane_mode_on=1`, API 35, x86_64 |
| Приватность и упаковка | проверка merged manifests, `git ls-files "*.onnx"`, список APK через `jar tf` | OK — `INTERNET` и `ACCESS_NETWORK_STATE` отсутствуют; 0 ONNX в Git, рабочем дереве и APK |

## 3. Изменённые файлы

- `.gitignore` — запрет добавления `*.onnx`.
- `gradle/libs.versions.toml`, `app/build.gradle.kts`, `app/proguard-rules.pro` — ONNX Runtime 1.26.0, test dependencies, parity assets и правила shrinker.
- `app/src/main/java/com/faceswaplocal/app/inference/FaceGeometry.kt` — FaceFusion templates, similarity transform и inverse transform.
- `app/src/main/java/com/faceswaplocal/app/inference/ModelCatalog.kt` — allowlist имён, размеров и SHA-256.
- `app/src/main/java/com/faceswaplocal/app/inference/ModelStore.kt` — приватный импорт, integrity state machine и атомарная замена.
- `app/src/main/java/com/faceswaplocal/app/inference/OnnxInitializerReader.kt` — потоковое извлечение InSwapper `emap` из ONNX.
- `app/src/main/java/com/faceswaplocal/app/inference/OnnxRawFaceSwapPipeline.kt` — detector, alignment, embedding, conversion и raw swapper inference.
- `app/src/main/java/com/faceswaplocal/app/ui/FaceSwapApp.kt`, `FaceSwapViewModel.kt` — UI импорта/выбора/сырого swap, coroutine orchestration и bitmap lifetime.
- `app/src/test/java/com/faceswaplocal/app/inference/*.kt` — 20 новых inference/geometry unit-тестов; вместе с 4 существующими доменными тестами — 24 на вариант.
- `app/src/androidTest/java/com/faceswaplocal/app/inference/FaceFusionParityInstrumentedTest.kt` — Android geometry/raw parity для 3 пар × 2 swapper.
- `docs/MODEL_CARD.md`, `THIRD_PARTY_NOTICES.md`, `PRIVACY.md` — модели, лицензии и локальная политика данных.
- `docs/BENCHMARKS.md`, `docs/KNOWN_LIMITATIONS.md` — измерения и ограничения этапа.
- `docs/parity/` — синтетические inputs, reference runner, FaceFusion reference tensors/crops/outputs, Android outputs и воспроизводимые команды.
- `docs/reports/img/STAGE_B_RAW_CROP_API35.png` — результат ручного UI-smoke на синтетике в авиарежиме.

## 4. Benchmark и parity (если предусмотрены этапом)

- Desktop reference: FaceFusion 3.7.1 (`3f81a8a78454089d720b8f318a12ae1702c4633b`), Python 3.12.10, ONNX Runtime 1.26.0 CPU, OpenCV 4.13.0, NumPy 2.2.1. Команда и окружение: `docs/parity/README.md`; runner: `docs/parity/run_facefusion_reference.py`.
- Android: AVD API 35 x86_64, ONNX Runtime 1.26.0, CPU. Максимальная ошибка landmarks — `0.123990 px`; минимальный SSIM source crop — `0.998169`; минимальный SSIM target crop — `0.997606`; минимальный raw output SSIM — `0.997513`.

| Пара | Swapper | Raw SSIM Android ↔ FaceFusion |
| --- | --- | ---: |
| `pair_01` | HyperSwap 1a 256 | 0.998060 |
| `pair_01` | InSwapper 128 fp16 | 0.998275 |
| `pair_02` | HyperSwap 1a 256 | 0.997760 |
| `pair_02` | InSwapper 128 fp16 | 0.997551 |
| `pair_03` | HyperSwap 1a 256 | 0.997513 |
| `pair_03` | InSwapper 128 fp16 | 0.997645 |

- Медиана Android instrumentation total: HyperSwap `22 777 ms`, InSwapper `39 766 ms`. Это AVD-измерения, не прогноз производительности ARM-телефона.
- Повторный ручной UI-прогон финального APK в авиарежиме: detector `11 326 ms`, ArcFace `18 611 ms`, InSwapper `154 649 ms`, total `185 719 ms`. Холодный UI-прогон включает повторный SHA-256 всех необходимых моделей; контролируемые parity timings приведены выше.
- Физическое Android-устройство не подключено. Модель телефона, SoC, RAM, peak Java/native heap и thermal state должны быть добавлены в `docs/BENCHMARKS.md`, когда появится первый reference device.

## 5. Отклонения от ТЗ

- Android использует детерминированное closed-form least-squares similarity решение вместо `cv2.estimateAffinePartial2D(..., RANSAC, 100)` и собственный bilinear sampler с edge clamp вместо OpenCV `INTER_AREA/BORDER_REPLICATE`. Причина — отсутствие OpenCV в Android-приложении; расхождение измерено parity-тестом и укладывается в `0.123990 px` / raw SSIM не ниже `0.997513`.
- Parity уровня геометрии сравнивает raw 5 точек YOLOFace на обеих сторонах. Полный desktop FaceFusion при другом режиме может уточнять геометрию через 68→5 landmarks; этот режим не использовался, чтобы проверить именно обязательный Stage B detector contract.
- HyperSwap остаётся первым кандидатом shortlist, но не выбран рабочим default: на всех трёх синтетических парах его output численно совпадает с FaceFusion, однако визуальный перенос identity слабый. Утверждённый fallback `inswapper_128_fp16` проходит и численную, и визуальную часть DoD.
- XNNPACK session path реализован, но ORT 1.26.0 на x86_64 AVD дал native `SIGABRT` в `OrtSession_createSession`, который невозможно перехватить Kotlin exception. Для x86/x86_64 введён безопасный предварительный `CPU_FALLBACK`; корректность XNNPACK на ARM64 не заявляется до проверки на реальном устройстве.
- Код FaceFusion в приложение не копировался: самостоятельно реализованы алгоритмы по tensor contracts, числовым шаблонам, порядку операций и проверенным выходам, поскольку upstream license-файл 3.7.1 не содержит полного текста названной OpenRAIL-AS лицензии.

## 6. Известные проблемы и ограничения

- Этап B выдаёт только raw crop. Inverse transform, маска, feathering, color matching и blending относятся к этапу C; GFPGAN и улучшенная occlusion mask — к этапу E.
- HyperSwap 1a требует дополнительной качественной проверки на лицензированном real-photo fixture или reference device; текущие synthetic fixtures не подтвердили достаточный перенос identity.
- XNNPACK/ARM64, NNAPI, peak heap, thermal throttling и повторные циклы памяти не проверены без физического reference device.
- Debug APK занимает `171.45 MiB` из-за нативных ABI-библиотек ONNX Runtime; весов моделей в APK нет. ABI splits/minification относятся к этапу G.
- Полная повторная SHA-256-проверка больших файлов перед каждой новой session увеличивает холодный запуск на медленном AVD; это сознательная integrity-гарантия §11.3.
- Условия HyperSwap `ResearchRAIL` не представлены upstream полным license-текстом; InsightFace weights имеют non-commercial research restriction; metadata YOLOFace содержит конфликтующие copyleft-обозначения. Поэтому никакие веса не распространяются приложением или репозиторием.

## 7. Блокеры

- Для CPU Definition of Done этапа B блокеров нет. Проверка XNNPACK/ARM64 и benchmark reference device ожидают физическое устройство и не заявлены как выполненные.

## 8. Следующий шаг

- После подтверждения пользователя перейти к этапу C: inverse transform, аффинная маска, feathering, color matching, мягкий blending, полный сценарий «один источник → одно целевое лицо», CPU fallback и parity финального кадра.
