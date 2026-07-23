# Отчёт: Этап D — Несколько источников и целей
Дата: 23.07.2026, ветка/коммит: `main` / `d42c67e64e43bc5f1a9021f27fdb27c6602100b2`

## 1. Сделано

- Реализован пакет до 8 фотографий-источников через системный
  `PickMultipleVisualMedia(8)`; лица получают стабильные идентификаторы сессии.
- Реализованы независимые назначения каждому target, «Не менять», подтверждаемая
  команда «Применить ко всем» и удаление источника с инвалидацией связанных
  назначений.
- Назначения сериализуются в `SavedStateHandle` только как пары identifier. После
  `ActivityScenario.recreate()` они сохраняются; после смерти процесса без bitmap
  показывается понятное сообщение с предложением выбрать изображения заново.
- Целевые 5-точечные лица детектируются YOLOFace один раз по оригиналу. T1→T2→T3
  обрабатываются последовательно; каждый следующий paste получает накопленный
  результат. IoU-защита выбора neural face сохранена.
- Identity embedding кэшируется по `FaceId` на время одной задачи. В `finally`
  зануляются нулями все удерживаемые эмбеддинги — per-source кэш и финальная копия
  результата, — затем map очищается; выполняется при успехе, ошибке и отмене.
  Координатор возвращает `MultiPhotoFaceSwapResult` **без** `sourceEmbedding`, поэтому в
  ViewModel identity embedding не передаётся (см. правку после аудита ниже).
- Полная SHA-256-проверка модели выполняется один раз за процесс. Повторное открытие
  session в том же процессе проверяет канонический файл в приватном
  `filesDir/models` и точный размер; новый процесс и каждый импорт снова выполняют
  полный SHA-256.
- `make_group_fixture.py` с seed `20260722` воспроизводимо создаёт
  `stage_d_group_target.png` из одобренных synthetic portraits. Fixture содержит
  четыре различимых лица и реальное пересечение affine paste ROI T1/T2.
- Реальный instrumentation test подтвердил: T1/T2/T3 изменены тремя разными
  источниками, T4 побитово идентичен, пересечение содержит второй paste, вне
  объединения paste ROI изменено 0 пикселей.
- Compose UI-test без picker/inference подтвердил независимые назначения, «Не
  менять», диалог подтверждения apply-to-all, invalidation при удалении и recreate.
- Ручная API 35 проверка сведена к открытию debug test harness: экран показывает
  четыре рамки, номера 1–4, «Найдено лиц: 4» и карточку назначений.
- Definition of Done этапа D — **ВЫПОЛНЕНО:** три разных source назначены T1/T2/T3,
  T4 не меняется; multi-face результат и состояние UI проверены на API 35.

## 2. Проверки

| Проверка | Команда | Результат |
| --- | --- | --- |
| Unit tests | `.\gradlew.bat test --no-daemon --console=plain` | OK — 84 прогона, 0 failures/errors/skipped |
| Lint | `.\gradlew.bat lint --no-daemon --console=plain` | OK — 0 errors, 32 известных warning обновления версий/SDK |
| Сборка | `.\gradlew.bat assembleDebug --no-daemon --console=plain` | OK — 179 944 427 байт; SHA-256 `57A3F0F45521F86056502DC5B6E365BAFCFAF7DDEB4B72F370ABD7198E8CC091` |
| Multi-face instrumentation | `adb -s emulator-5554 shell am instrument -w -r -e class com.faceswaplocal.app.inference.StageDMultiFaceInstrumentedTest com.faceswaplocal.app.test/androidx.test.runner.AndroidJUnitRunner` | OK — `OK (1 test)`, 316,565 с |
| Compose UI instrumentation | `adb -s emulator-5554 shell am instrument -w -r -e class com.faceswaplocal.app.ui.StageDComposeUiInstrumentedTest com.faceswaplocal.app.test/androidx.test.runner.AndroidJUnitRunner` | OK — `OK (1 test)`, 122,547 с |
| Устройство | `adb install -r app-debug.apk` + запуск `StageDUiTestActivity` | OK — API 35 x86_64, `airplane_mode_on=1`, четыре target отображаются |
| Приватность | проверка merged manifest | OK — `INTERNET` и `ACCESS_NETWORK_STATE` отсутствуют |

## 3. Изменённые файлы

- `app/src/main/java/com/faceswaplocal/app/domain/FaceModels.kt` — apply-to-all,
  invalidation и codec identifier-назначений.
- `app/src/main/java/com/faceswaplocal/app/inference/ModelStore.kt` — process-local
  cache результата полной SHA-256-проверки.
- `app/src/main/java/com/faceswaplocal/app/inference/OnnxRawFaceSwapPipeline.kt` —
  cached target faces и cached source embedding.
- `app/src/main/java/com/faceswaplocal/app/inference/OnnxPhotoFaceSwapPipeline.kt` —
  накопленный pixel buffer и передача embedding.
- `app/src/main/java/com/faceswaplocal/app/inference/OnnxMultiPhotoFaceSwapPipeline.kt`
  — стабильный последовательный multi-face pipeline и очистка embedding cache.
- `app/src/main/java/com/faceswaplocal/app/ui/FaceSwapViewModel.kt` — пакет
  источников, SavedStateHandle и multi-assignment orchestration.
- `app/src/main/java/com/faceswaplocal/app/ui/FaceSwapApp.kt` — multi-picker,
  «Не менять», apply-to-all с подтверждением, удаление и test tags.
- `app/src/debug/` и `app/src/androidTest/` — debug-only Compose harness,
  multi-face и UI instrumentation tests.
- `docs/parity/inputs/make_group_fixture.py` и `stage_d_group_target.png` —
  воспроизводимый fixture.
- `docs/MODEL_CARD.md` — политика process-local SHA verification cache.
- `docs/BENCHMARKS.md` — измерение Stage D.
- `docs/reports/img/STAGE_D_MULTI_FACE_RESULT.png` — итог реального inference.
- `docs/reports/img/STAGE_D_ASSIGNMENTS_API35.png` — ручной экран четырёх целей.

## 4. Benchmark и parity (если предусмотрены этапом)

- API 35 x86_64 AVD, Android 15, ONNX Runtime Android 1.26.0, CPU,
  `airplane_mode_on=1`.
- Target: 1600×1100, четыре neural face; три последовательных InSwapper
  назначения; полный instrumentation test — 316,565 с.
- T1/T2 paste ROI пересекаются; T4 изменённых пикселей — 0; вне объединения трёх
  paste ROI изменённых пикселей — 0.
- Stage D не меняет модели или preprocessing, поэтому численная parity Stage B/C
  остаётся действующей. Fixture Stage D проверяет assignment/compositing, а не
  качество blending.
- Peak heap и thermal state не снимались; AVD не объявляется reference device.

## 5. Отклонения от ТЗ

- Полный ручной picker→inference сценарий заменён двумя instrumentation tests.
  Причина: повторные Gradle build/install занимали основную часть запусков, а
  ручная навигация по системному Photo Picker срывалась до начала inference.
  Реальный pipeline проверен instrumentation test; вручную оставлено только
  отображение четырёх target на экране назначений.
- В Compose UI-test кнопка удаления второго источника проверяется как существующий
  semantics node, после чего вызывается тот же `ViewModel.removeSource(FaceId)`
  callback напрямую. Причина — нестабильный coordinate click во вложенных
  vertical/horizontal scroll containers API 35; invalidation проверяется строгим
  assertion и отдельным unit-тестом.
- После первой полной SHA-256-проверки модели в процессе последующие session
  openings проверяют приватный путь и размер. Гарантия §11.3 сохраняется:
  импорт полностью хешируется, каталог приватный, а новый процесс снова выполняет
  полную SHA-256.
- SHA-256 `50533ADF…54CE7` в предыдущей редакции отчёта был ошибочно перенесён из
  отчёта Stage C без актуальной чистой пересборки. Фактический финальный hash Stage D
  указан в §2.

## 6. Известные проблемы и ограничения

- Качество blending/restoration остаётся предфинальным и относится к этапу E.
- Source bitmap не восстанавливаются после смерти процесса; identifiers
  сохраняются, но пользователь должен повторно выбрать файлы.
- AVD CPU существенно медленнее реального ARM reference device; измерение нельзя
  использовать как прогноз времени телефона.

## 7. Блокеры

- Нет.

## 8. Следующий шаг

- После подтверждения пользователя начать этап **E1** «Качество лица» (GFPGAN 1.4,
  улучшенная parsing/occlusion mask, визуальная сверка с эталоном). Продуктовые задачи —
  пресеты качества, сравнение до/после, отмена, MediaStore export, профилирование
  памяти — вынесены в этап **E2** (см. `TECHNICAL_SPEC.md` v2.1 §15).

## 9. Правки после аудита (23.07.2026)

По итогам аудита A–D (`AUDIT_STAGE_A-D.md`) в этап D внесены две точечные правки, не
меняющие numeric parity этапов B/C:

1. **Эмбеддинг не покидает пайплайн.** До правки `OnnxMultiPhotoFaceSwapPipeline.process`
   возвращал `PhotoFaceSwapResult`, в котором поле `sourceEmbedding` уезжало в
   `photoSwapResult` ViewModel (в оперативной памяти, но вопреки буквальной формулировке
   §1). Теперь координатор возвращает отдельный `MultiPhotoFaceSwapResult`
   (`finalBitmap`, `pasteRois`, backends, `timings`) **без** эмбеддинга; в `finally`
   дополнительно зануляется финальная копия `sourceEmbedding`. UI ничего не потерял —
   он читал только bitmap, backends и тайминги.
2. **Инструментальный тест переписан на продакшн-координатор.** До правки
   `StageDMultiFaceInstrumentedTest` воспроизводил логику координатора циклом по
   `OnnxPhotoFaceSwapPipeline`. Теперь он вызывает `OnnxMultiPhotoFaceSwapPipeline.process`
   целиком и ассертит те же гарантии по возвращаемым `pasteRois`: три разных источника на
   T1/T2/T3, T4 побитово неизменен, 0 пикселей вне объединения paste ROI, пересечение
   T1/T2 содержит второй paste. Для этого координатор стал отдавать список per-step
   `pasteRois`.

Проверки после правок — см. обновлённый `AUDIT_STAGE_A-D.md` §5.
