# Отчёт: Этап D — Несколько источников и целей
Дата: 22.07.2026, ветка/коммит: main / e49cee0dcec177f01dbffbf892fcdcc91f94b010

## 1. Сделано
- Пакет источников до 8 фотографий — `FaceSwapApp.kt` использует системный `PickMultipleVisualMedia(8)`, а `FaceSwapViewModel.kt` декодирует и детектирует лица каждой выбранной фотографии локально; лица получают стабильные session-id `source-<photo>-<face>`.
- Выбор лиц и назначения — `FaceAssignmentPlanner` и Compose-карточка поддерживают независимые назначения каждому target, «Не менять», удаление источника и подтверждаемую команду «Применить ко всем».
- Invalidated assignments — удаление источника удаляет все назначения, зависящие от его `FaceId`; это покрыто unit-тестом.
- Последовательный multi-face pipeline — `OnnxMultiPhotoFaceSwapPipeline.kt` один раз вызывает YOLOFace для исходного target, сохраняет полученные 5-точечные лица и затем в стабильном порядке передаёт предыдущие composite pixels следующему paste. Геометрия и 5-точечная детекция всегда относятся к неизменённому target.
- IoU-защита выбора сохранена: каждый source/target hint по-прежнему выбирает только пересекающегося YOLO-кандидата с максимальным IoU.
- Overlap-проверка — `FaceCompositorTest` моделирует близко расположенные, пересекающиеся face ROI и проверяет, что область первого лица сохраняется, а пересечение содержит второй composite, а не восстановленный оригинал.
- Добавлены `docs/parity/inputs/make_group_fixture.py` (seed `20260722`) и versioned `stage_d_group_target.png`: четыре различимых synthetic лица, T1/T2 с пересекающимися paste ROI. `STAGE_D_GROUP_FIXTURE_CHECKLIST.md` явно ограничивает назначение fixture логикой назначений/compositing, а не качеством blending.
- SavedStateHandle хранит только пары identifier target/source; при process death UI сообщает, что изображения надо выбрать заново, а при повторной детекции валидные identifiers восстанавливают назначения.
- Definition of Done «минимум три разных источника трём людям на одном фото» — не выполнен вручную: fixture создан, APK установлен в авиарежиме, но полный picker→3→3→rotate прогон не выполнен до окончания запуска.

## 2. Проверки
| Проверка | Команда | Результат |
| --- | --- | --- |
| Unit tests | `./gradlew.bat test --console=plain` | OK — 41 тест в debug и 41 в release, 82 успешных прогона |
| Lint | `./gradlew.bat lint --console=plain` | OK — ошибок нет |
| Сборка | `./gradlew.bat assembleDebug --console=plain` | OK — `app-debug.apk`, SHA-256 `50533ADF8F3496235C7A8E78A8C476B5B47CB6E21BE27EA1990B7A1089854CE7` |
| Устройство | `adb -s emulator-5554 install -r app/build/outputs/apk/debug/app-debug.apk` | OK — APK установлен на API 35 x86_64, `airplane_mode_on=1`; сохранён `docs/reports/img/STAGE_D_API35_AFTER_INSTALL.png`; полный ручной 3→3/rotate не выполнен |

## 3. Изменённые файлы
- `app/src/main/java/com/faceswaplocal/app/domain/FaceModels.kt` — правила «не менять», apply-to-all и invalidated assignments.
- `app/src/main/java/com/faceswaplocal/app/inference/OnnxRawFaceSwapPipeline.kt` — передача один раз рассчитанных target face candidates.
- `app/src/main/java/com/faceswaplocal/app/inference/OnnxPhotoFaceSwapPipeline.kt` — поддержка cached target detection и накопленного pixel buffer.
- `app/src/main/java/com/faceswaplocal/app/inference/OnnxMultiPhotoFaceSwapPipeline.kt` — последовательный orchestrator нескольких лиц.
- `app/src/main/java/com/faceswaplocal/app/ui/FaceSwapApp.kt` — multiple-photo picker, per-target «Не менять», confirmation apply-to-all и удаление источника.
- `app/src/main/java/com/faceswaplocal/app/ui/FaceSwapViewModel.kt` — локальный пакет source bitmap/face и multi-assignment запуск.
- `app/src/test/java/com/faceswaplocal/app/domain/FaceAssignmentPlannerTest.kt` — invalidated assignments и apply-to-all.
- `app/src/test/java/com/faceswaplocal/app/inference/FaceCompositorTest.kt` — тест последовательного compositing пересекающихся ROI.
- `docs/parity/inputs/make_group_fixture.py`, `stage_d_group_target.png`, `STAGE_D_GROUP_FIXTURE_CHECKLIST.md` — воспроизводимый synthetic group fixture и границы его проверки.

## 4. Benchmark и parity (если предусмотрены этапом)
- Этап D не добавляет новую модель и не меняет Stage C parity. CPU fallback передаётся в каждый шаг multi-face pipeline; целевая YOLOFace-детекция выполняется ровно один раз на неизменённом bitmap до цикла compositing.
- Производительный benchmark и визуальная проверка трёх реальных назначений не выполнены: отсутствует разрешённый групповой input fixture.

## 5. Отклонения от ТЗ
- Ручной критерий «три разных источника трём людям на одном фото» и проверка поворота/recreate не закрыты в этом запуске; fixture уже добавлен, но UI-прогон не завершён.

## 6. Известные проблемы и ограничения
- Decoded bitmap намеренно не сохраняется после process death; вместо пустого экрана показывается понятное сообщение и требуется повторный picker.

## 7. Блокеры
- Выполнить ручной API 35 сценарий fixture: три разных источника → T1/T2/T3, T4 «Не менять», затем rotate/recreate и сохранить до/после.

## 8. Следующий шаг
- Предоставить или утвердить лицензированный/synthetic group fixture и завершить оставшуюся ручную проверку Stage D; после её успешного результата обновить этот отчёт и только затем переходить к этапу E.
