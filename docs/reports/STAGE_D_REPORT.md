# Отчёт: Этап D — Несколько источников и целей
Дата: 22.07.2026, ветка/коммит: main / 4a4bced (коммит до отчёта)

## 1. Сделано
- Пакет источников до 8 фотографий — `FaceSwapApp.kt` использует системный `PickMultipleVisualMedia(8)`, а `FaceSwapViewModel.kt` декодирует и детектирует лица каждой выбранной фотографии локально; лица получают стабильные session-id `source-<photo>-<face>`.
- Выбор лиц и назначения — `FaceAssignmentPlanner` и Compose-карточка поддерживают независимые назначения каждому target, «Не менять», удаление источника и подтверждаемую команду «Применить ко всем».
- Invalidated assignments — удаление источника удаляет все назначения, зависящие от его `FaceId`; это покрыто unit-тестом.
- Последовательный multi-face pipeline — `OnnxMultiPhotoFaceSwapPipeline.kt` один раз вызывает YOLOFace для исходного target, сохраняет полученные 5-точечные лица и затем в стабильном порядке передаёт предыдущие composite pixels следующему paste. Геометрия и 5-точечная детекция всегда относятся к неизменённому target.
- IoU-защита выбора сохранена: каждый source/target hint по-прежнему выбирает только пересекающегося YOLO-кандидата с максимальным IoU.
- Overlap-проверка — `FaceCompositorTest` моделирует близко расположенные, пересекающиеся face ROI и проверяет, что область первого лица сохраняется, а пересечение содержит второй composite, а не восстановленный оригинал.
- Definition of Done «минимум три разных источника трём людям на одном фото» — не выполнен: в текущем parity-наборе нет лицензированного группового фото с тремя лицами, поэтому на устройстве этот ручной сценарий не заявляется проверенным.

## 2. Проверки
| Проверка | Команда | Результат |
| --- | --- | --- |
| Unit tests | `./gradlew.bat test --console=plain` | OK — 41 тест в debug и 41 в release, 82 успешных прогона |
| Lint | `./gradlew.bat lint --console=plain` | OK — ошибок нет |
| Сборка | `./gradlew.bat assembleDebug --console=plain` | OK — `app-debug.apk`, SHA-256 `50533ADF8F3496235C7A8E78A8C476B5B47CB6E21BE27EA1990B7A1089854CE7` |
| Устройство | `adb -s emulator-5554 install -r app/build/outputs/apk/debug/app-debug.apk` | OK — APK установлен и запущен на API 35 x86_64; ручной сценарий 3→3 не выполнялся: отсутствует групповой fixture |

## 3. Изменённые файлы
- `app/src/main/java/com/faceswaplocal/app/domain/FaceModels.kt` — правила «не менять», apply-to-all и invalidated assignments.
- `app/src/main/java/com/faceswaplocal/app/inference/OnnxRawFaceSwapPipeline.kt` — передача один раз рассчитанных target face candidates.
- `app/src/main/java/com/faceswaplocal/app/inference/OnnxPhotoFaceSwapPipeline.kt` — поддержка cached target detection и накопленного pixel buffer.
- `app/src/main/java/com/faceswaplocal/app/inference/OnnxMultiPhotoFaceSwapPipeline.kt` — последовательный orchestrator нескольких лиц.
- `app/src/main/java/com/faceswaplocal/app/ui/FaceSwapApp.kt` — multiple-photo picker, per-target «Не менять», confirmation apply-to-all и удаление источника.
- `app/src/main/java/com/faceswaplocal/app/ui/FaceSwapViewModel.kt` — локальный пакет source bitmap/face и multi-assignment запуск.
- `app/src/test/java/com/faceswaplocal/app/domain/FaceAssignmentPlannerTest.kt` — invalidated assignments и apply-to-all.
- `app/src/test/java/com/faceswaplocal/app/inference/FaceCompositorTest.kt` — тест последовательного compositing пересекающихся ROI.

## 4. Benchmark и parity (если предусмотрены этапом)
- Этап D не добавляет новую модель и не меняет Stage C parity. CPU fallback передаётся в каждый шаг multi-face pipeline; целевая YOLOFace-детекция выполняется ровно один раз на неизменённом bitmap до цикла compositing.
- Производительный benchmark и визуальная проверка трёх реальных назначений не выполнены: отсутствует разрешённый групповой input fixture.

## 5. Отклонения от ТЗ
- Ручной критерий «три разных источника трём людям на одном фото» и проверка поворота/recreate не закрыты. Причина: в `docs/parity/inputs/` есть только одиночные портреты; добавление нового группового изображения без проверенной лицензии/происхождения не выполнялось.

## 6. Известные проблемы и ограничения
- UI пока хранит decoded source bitmaps только в памяти активной ViewModel; стандартный поворот сохраняет ViewModel, но process recreation требует повторного выбора файлов.
- Для нескольких лиц каждый source embedding пока вычисляется отдельно для каждого назначения; session resources закрываются между шагами. Кэширование embedding остаётся оптимизацией следующего этапа качества/производительности.

## 7. Блокеры
- Для закрытия Stage D нужен синтетический или явно лицензированный групповой fixture с минимум четырьмя близко расположенными лицами. После добавления: выполнить ручной API 35 сценарий 3 разных источника → 3 target, оставить четвёртый «Не менять», сделать rotate/recreate и записать результат.

## 8. Следующий шаг
- Предоставить или утвердить лицензированный/synthetic group fixture и завершить оставшуюся ручную проверку Stage D; после её успешного результата обновить этот отчёт и только затем переходить к этапу E.
