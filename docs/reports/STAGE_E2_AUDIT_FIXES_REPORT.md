# Отчёт: Этап E2 — аудиторские правки (AUDIT_STAGE_E1-E2.md §2.1, §2.2, §2.4)
Дата: 09.08.2026, ветка/коммит: `main` / `0271996`

Это не новая контрольная точка этапа E2 — контрольная точка 3 (пресеты качества, сила
идентичности, ручная растушёвка, автопереключатель цветокоррекции) не начиналась. Этот
отчёт закрывает три пункта из `docs/reports/AUDIT_STAGE_E1-E2.md` §3 («Что требует
исправления»): асимметрию защиты неназначенных лиц (§2.2), отсутствие ассерта на
занулённость эмбеддингов (§2.1) и отсутствие теста на отмену внутри прохода
восстановления (§2.4).

## 1. Сделано

- **(§2.2) Свап-проход защищает неназначенные лица тем же parser-механизмом, что и
  восстановление, но платит за это только при геометрическом риске.**
  `OnnxMultiPhotoFaceSwapPipeline.kt` до начала прохода 1 вычисляет для каждого
  назначенного лица набор «зон риска» — `swapCropRoi` (реальный paste ROI свапа,
  геометрия `ARCFACE_128`/128×128, без инференса) и, если восстановление запланировано,
  `ffhqRoi` (FFHQ-512 ROI восстановления). Неназначенное лицо получает
  face-shaped parser-маску (`protectedFaceRegions`, BiSeNet, `PERSON_CLASS_IDS`) или, в
  режиме без парсера, жёсткий прямоугольник (`protectedBaseRois`) **только если** его
  FFHQ ROI пересекается (`CompositeRoi.intersects`, `FaceCompositor.kt`) хотя бы с одной
  зоной риска. Для геометрически далёких лиц список защиты остаётся пустым: ни
  дополнительного инференса BiSeNet, ни лишней записи в `pasteBack` не происходит —
  цена платится только там, где есть за что платить. Публичное поле
  `MultiPhotoFaceSwapResult.protectedUnassignedRois` не сужено: оно по-прежнему содержит
  ROI всех неназначенных лиц (это отчётное поле, используемое как контрфактическая
  проверка в `StageETwoPassCoordinatorInstrumentedTest`, а не поле, управляющее
  инференсом).
- **Честная оговорка о статусе этой правки записана в самом аудите
  (`AUDIT_STAGE_E1-E2.md` §2.2, блок «Обновление 09.08.2026»).** Правка — защита в
  глубину (defence in depth) на основании кода и геометрического рассуждения, а не
  исправление подтверждённого дефекта: `StageEDenseUnassignedFaceInstrumentedTest` не
  воспроизводит регрессию на дереве до правки — фикстура и оба теста добавлены вместе с
  правкой, а не как повтор упавшего прогона, и ни один существующий тест на плотной
  геометрии не падал. Одновременно в аудит добавлен отдельный **непроверенный** пункт:
  два физически перекрывающихся человека, где сегментация неназначенного лица может
  ошибочно разметить пиксели соседнего назначенного человека как «кожа»/«волосы».
  Подходящей фикстуры для этого случая в репозитории нет; создание такой фикстуры не
  входило в эту задачу, и пункт явно помечен как нерешённый.
- **(§2.1) Занулённость эмбеддингов теперь проверяется ассертом, а не только чтением
  кода.** Добавлен `EmbeddingLifecycleListener` (`EmbeddingLifecycle.kt`) — диагностический
  хук, устроенный по образцу уже существующего `InferenceSessionLifecycleListener`:
  no-op в продакшене, в тестах фиксирует ссылку на каждый массив эмбеддинга, который
  координатор позже занулит в `finally`, включая копии, отклонённые `putIfAbsent`
  (ровно то место, что уже один раз ломалось — см. §1.4 аудита).
  `StageEEmbeddingHygieneInstrumentedTest` проверяет два сценария: занулённость после
  успешного прогона и после отменённого прогона, причём предварительно доказывает, что
  захваченные массивы не были нулевыми уже в момент захвата — иначе финальная проверка
  была бы вакуумной.
- **(§2.4) Отмена внутри прохода восстановления покрыта реальным прогоном.**
  `StageE2RestorationCancellationInstrumentedTest` зеркалит
  `StageE2CancellationInstrumentedTest`, но отменяет выполнение после того, как первое
  лицо уже прошло восстановление (GFPGAN открывалась и использовалась), и до начала
  восстановления второго. Проверяется: свап-проход полностью завершился до первого
  события `RESTORING` (барьер проходов соблюдён), GFPGAN действительно открывалась
  (`listener.events` содержит `open:gfpgan_1.4.onnx`), и ни одна тяжёлая сессия не
  осталась открытой к моменту отмены.
- Плотная фикстура `stage_e_dense_pair_target.png` и оба её теста
  (`StageEDenseUnassignedFaceInstrumentedTest`) оставлены как регрессия: они
  единственные в дереве доказывают, что защита реально включается на пересекающейся
  геометрии, а не только на геометрии, где она и не нужна.

## 2. Проверки

| Проверка | Команда | Результат |
| --- | --- | --- |
| Unit tests + Lint + сборка debug/androidTest | `.\gradlew.bat test lint assembleDebug assembleDebugAndroidTest --no-daemon --console=plain` | `BUILD SUCCESSFUL` за 5 мин 27 с |
| Компиляция main + androidTest Kotlin (промежуточная проверка) | `.\gradlew.bat compileDebugKotlin compileDebugAndroidTestKotlin --no-daemon --console=plain` | `BUILD SUCCESSFUL` за 2 мин 12 с |
| Занулённость эмбеддингов (новый) | `adb shell am instrument ... -e class com.faceswaplocal.app.inference.StageEEmbeddingHygieneInstrumentedTest` | **OK (2 tests)**, 150,713 с |
| Отмена внутри восстановления (новый) | `adb shell am instrument ... -e class com.faceswaplocal.app.inference.StageE2RestorationCancellationInstrumentedTest` | **OK (1 test)**, 218,817 с |
| Плотная фикстура — «Не менять» в обоих проходах | `adb shell am instrument ... -e class com.faceswaplocal.app.inference.StageEDenseUnassignedFaceInstrumentedTest` | **OK (2 tests)**, 327,606 с |
| Регрессия: T4 контрфактический ROI, барьер проходов | `adb shell am instrument ... -e class com.faceswaplocal.app.inference.StageETwoPassCoordinatorInstrumentedTest` | **OK (1 test)**, 705,268 с |
| Регрессия: пары сессий парсера при свапе и восстановлении | `adb shell am instrument ... -e class com.faceswaplocal.app.inference.StageESwapParserMaskInstrumentedTest` | **OK (2 tests)**, 318,783 с |
| Устройство | `adb install -r app-debug.apk` + `app-debug-androidTest.apk` на `emulator-5554`, API 35 | `Success` (оба) |

Каждый инструментальный класс запускался **по отдельности**: параллельный запуск двух
классов на одном эмуляторе один раз вызвал `force-stop` общего процесса
(`reason=10 USER REQUESTED subreason=21 FORCE STOP ... due to start instr`) — тот же
эффект, что уже описан в `STAGE_E2_CHECKPOINT_2_REPORT.md`. Это ошибка запуска, а не
падение теста; повторный одиночный прогон прошёл штатно.

`StageE2ExportInstrumentedTest`, `StageE2CancellationInstrumentedTest` и остальная
регрессия E1/E2, не связанная с защитой неназначенных лиц или эмбеддингами, повторно не
прогонялась — код путей экспорта, водяного знака и EXIF этой правкой не затронут.

## 3. Изменённые файлы

Новые:

- `app/src/main/java/com/faceswaplocal/app/inference/EmbeddingLifecycle.kt` —
  `EmbeddingLifecycleListener` и no-op реализация по умолчанию.
- `app/src/androidTest/java/com/faceswaplocal/app/inference/StageEEmbeddingHygieneInstrumentedTest.kt`
  — занулённость эмбеддингов после успеха и после отмены.
- `app/src/androidTest/java/com/faceswaplocal/app/inference/StageE2RestorationCancellationInstrumentedTest.kt`
  — отмена внутри прохода восстановления, GFPGAN открыта-и-закрыта.
- `app/src/androidTest/java/com/faceswaplocal/app/inference/StageEDenseUnassignedFaceInstrumentedTest.kt`
  — «Не менять» на пересекающейся геометрии, оставлено как регрессия.
- `docs/parity/inputs/stage_e_dense_pair_target.png` — фикстура для теста выше.

Изменённые:

- `app/src/main/java/com/faceswaplocal/app/inference/OnnxMultiPhotoFaceSwapPipeline.kt`
  — геометрический гейт риска (`swapCropRoi`, `assignedRiskRois`,
  `riskyUnassignedFaces`), условный вызов `protectedFaceRegions`, вызов
  `embeddingLifecycle.onEmbeddingProduced`.
- `app/src/main/java/com/faceswaplocal/app/inference/FaceCompositor.kt` —
  `CompositeRoi.intersects`.
- `app/src/main/java/com/faceswaplocal/app/inference/OnnxPhotoFaceSwapPipeline.kt`,
  `OnnxFaceEnhancerPipeline.kt`, `OnnxFaceParserPipeline.kt` — проброс
  `protectedBaseRois`/`protectedFaceRegions` до `pasteBack` и `classIds` до
  `createRegionMask` (`PERSON_CLASS_IDS` для защиты, `REGION_CLASS_IDS` для блендинга
  свапа — без изменений поведения).
- `docs/parity/inputs/make_group_fixture.py` — генератор плотной фикстуры
  (`dense_pair()`), генератор `stage_d_group_target.png` не тронут (его байты
  зафиксированы для существующих parity-артефактов).
- `docs/reports/AUDIT_STAGE_E1-E2.md` — §2.2 понижена до «проверено, риск не
  материализуется на непересекающихся лицах» с явной оговоркой о статусе правки;
  добавлен непроверенный пункт про физически перекрывающихся людей; §3 п. 1 отмечен
  как сделанный со ссылкой на обновление.

## 4. Benchmark и parity

Не предусмотрены этой правкой: числовые пути свапа, блендинга и восстановления не
менялись — менялось только то, при каком условии вызывается уже существующий код
защиты. Отдельного замера «стоимости, которая теперь не платится» на далёкой геометрии
не делалось; структурная гарантия — пустой список защиты означает ноль дополнительных
вызовов `createRegionMask` и ноль дополнительных элементов в цикле `pasteBack` — видна
из кода и не требует профилирования, чтобы быть корректной, но не измерена в миллисекундах.

## 5. Отклонения от ТЗ

- **`EmbeddingLifecycleListener`** — DI-хук, отсутствующий в ТЗ. Он тестовый по
  назначению (в продакшене всегда no-op) и сделан по образцу уже принятого в проекте
  `InferenceSessionLifecycleListener`; без него утверждение «эмбеддинги зануляются»
  оставалось бы непроверяемым чёрным ящиком.

## 6. Известные проблемы и ограничения

- Сценарий из нового пункта аудита §2.2 — физически перекрывающиеся люди, где парсер
  размечает соседа как кожу — остаётся непроверенным. Нужна фикстура с реальным
  перекрытием лиц; её создание не входило в эту задачу.
- §2.3 аудита (защита зависит от детектора YOLOFace) не менялась и остаётся
  архитектурным ограничением.
- Оставшийся пункт §2.4 аудита — `deleteStagingFiles()` чистит весь `cacheDir/export`,
  что станет проблемой только при появлении параллельного экспорта — не менялся, риск
  сегодня недостижим.
- Контрольная точка 3 этапа E2 (пресеты качества, сила идентичности, ручная
  растушёвка, автопереключатель цветокоррекции) не начиналась.

## 7. Блокеры

- Нет.

## 8. Следующий шаг

- Фикстура с физически перекрывающимися людьми, чтобы закрыть последний непроверенный
  пункт §2.2, либо контрольная точка 3 этапа E2 (пресеты качества и сила идентичности) —
  на усмотрение пользователя, обе задачи независимы друг от друга.
