# Отчёт: Этап E2, контрольная точка 1 — экспорт и прогресс с отменой
Дата: 03.08.2026, ветка/коммит: `main` / `61f4b4f`

Этап E2 целиком не закрыт. Эта контрольная точка закрывает только два пункта задач
этапа из §15: экспорт через `MediaStore` (FR-PHOTO-09) и прогресс с отменой и cleanup
(FR-PHOTO-07). Пресеты качества, identity strength, ручной feathering, color auto/off
и профилирование памяти не начинались.

## 1. Сделано

- **Экспорт через `MediaStore` (FR-PHOTO-09)** — `app/src/main/java/com/faceswaplocal/app/data/ResultExporter.kt`.
  Результат кодируется в приватный staging-файл `cacheDir/export/export_*.part`, туда
  же пишется отметка о редактировании, и только затем поток копируется в назначение.
  На API 29+ назначение — строка `MediaStore` с `IS_PENDING = 1`,
  `RELATIVE_PATH = Pictures/FaceSwapLocal`; `IS_PENDING` снимается лишь после полной
  успешной записи. Разрешение полного доступа к хранилищу не запрашивается: в merged
  manifest нет ни `WRITE_EXTERNAL_STORAGE`, ни `READ_MEDIA_IMAGES`.
- **Имя файла** — `app/src/main/java/com/faceswaplocal/app/data/ExportNaming.kt`.
  `FaceSwapLocal_yyyyMMdd_HHmmss.ext` формируется по фиксированному шаблону с
  `Locale.ROOT` и ISO-хронологией, поэтому буддийский или исламский календарь
  устройства не меняет цифры имени.
- **Формат и качество** — `app/src/main/java/com/faceswaplocal/app/domain/ExportSettings.kt`.
  JPEG по умолчанию с настраиваемым качеством `60…100` (default `95`), PNG как опция.
  PNG получает lossless-качество независимо от запомненного положения ползунка.
- **Размеры результата** — экспортируемый файл имеет ровно те размеры, которые вернул
  пайплайн для обработанной цели; это проверяется и по строке `MediaStore`
  (`WIDTH`/`HEIGHT`), и по фактически декодированному файлу.
- **Успешное сохранение в UI** — `app/src/main/java/com/faceswaplocal/app/ui/FaceSwapApp.kt`:
  миниатюра результата, имя файла, альбом `FaceSwapLocal` с размерами и системная
  кнопка «Открыть» (`ACTION_VIEW` с `FLAG_GRANT_READ_URI_PERMISSION`).
- **Удаление незавершённой записи** — при ошибке кодирования, ошибке записи и отмене
  `finally` удаляет staging-файл (вместе с возможным `.tmp` от `ExifInterface`) и
  назначение, которое создал этот экспорт. Для `MediaStore` используется
  `ContentResolver.delete`, для SAF — `DocumentsContract.deleteDocument`.
- **Исходный файл не изменяется** — экспорт всегда создаёт новую запись и никогда не
  открывает выбранный пользователем файл на запись. Инструментальный тест сравнивает
  SHA-256 исходного JPEG до и после каждого сценария экспорта.
- **Метаданные проверены, а не предположены (§5.1)** — фикстура инструментального
  теста содержит настоящие теги GPS, `Make` и `DateTimeOriginal`. После перекодирования
  из bitmap в экспортированном файле `getLatLong` возвращает `false`, а `TAG_GPS_*`,
  `TAG_MAKE` и `TAG_DATETIME_ORIGINAL` отсутствуют. Дополнительно эти поля явно
  затираются как вторая линия защиты.
- **Нейтральная отметка о редактировании (§5.3)** — для JPEG в EXIF пишутся
  `Software = FaceSwapLocal` и `ImageDescription = Edited image`; для PNG добавляется
  чанк `tEXt` с ключевым словом `Software`
  (`app/src/main/java/com/faceswaplocal/app/data/PngEditMarker.kt`). Чанк дописывается
  на месте: `IEND` усекается, вставляется `tEXt`, `IEND` возвращается, поэтому
  полноразмерный PNG не переписывается через второй временный файл.
- **Видимый водяной знак (§5.3)** — отдельная настройка, включённая по умолчанию, с
  возможностью отключения (`app/src/main/java/com/faceswaplocal/app/data/ResultWatermark.kt`).
  Знак рисуется только в копии, уходящей в файл; bitmap результата, который
  пользователь сравнивает в режиме «До/После», не изменяется.
- **Прогресс (FR-PHOTO-07)** — `app/src/main/java/com/faceswaplocal/app/domain/ProcessingProgress.kt`
  и `onProgress` в `OnnxMultiPhotoFaceSwapPipeline`. Каждому этапу принадлежит
  фиксированный отрезок шкалы, счётчик лиц двигается только внутри своего отрезка, доля
  не убывает. UI показывает этап, номер текущего лица и общее количество; оставшееся
  время не обещается (§9.4). Считаются только назначенные лица.
- **Блокировка повторного запуска** — `canRunPhotoSwap` учитывает `RUNNING`,
  `CANCELLING` и выполняющийся экспорт; кнопка запуска и все настройки качества
  заблокированы во время обработки.
- **Отмена на безопасной границе** — кнопка «Отменить» переводит состояние в
  `CANCELLING` и отменяет job; координатор останавливается на ближайшем
  `ensureActive`, его собственный `finally` зануляет эмбеддинги, освобождает bitmap и
  закрывает сессии. Монотонный `photoSwapRunId` не даёт отменённому прогону перезаписать
  состояние уже начатого нового прогона.
- **Поворот экрана не перезапускает обработку** — job живёт в `viewModelScope`,
  прогресс и фаза читаются из `StateFlow`; инструментальный тест пересоздаёт Activity
  во время `RESTORING` и проверяет, что фаза, этап и номер лица сохранились.
- **Временные файлы только в приватном `cacheDir`** — каталог `cacheDir/export`
  принадлежит исключительно экспортёру. Очистка выполняется после успеха, ошибки и
  отмены, после каждого прогона обработки и при старте `ViewModel`. Стартовая уборка
  дополнительно удаляет строки `MediaStore`, оставленные этим приложением в состоянии
  `IS_PENDING` после аварийного завершения.

## 2. Проверки

| Проверка | Команда | Результат |
| --- | --- | --- |
| Unit tests | `.\gradlew.bat test --no-daemon --console=plain` | OK — 81 теста в debug и 81 в release, 0 failures/errors/skipped |
| Lint | `.\gradlew.bat lint --no-daemon --console=plain` | OK — 0 errors, 33 warnings |
| Сборка | `.\gradlew.bat assembleDebug --no-daemon --console=plain` | OK — `app-debug.apk`, 180 397 207 байт |
| Экспорт (инструментальный) | `adb shell am instrument ... -e class com.faceswaplocal.app.data.StageE2ExportInstrumentedTest` | OK — 7 tests, 2,757 с |
| Отмена координатора | `adb shell am instrument ... -e class com.faceswaplocal.app.inference.StageE2CancellationInstrumentedTest` | OK — 1 test, 82,294 с |
| Прогресс/экспорт в UI | `adb shell am instrument ... -e class com.faceswaplocal.app.ui.StageE2ProgressUiInstrumentedTest` | OK — 5 tests, 103,080 с |
| Регрессия E1 (два прохода) | `adb shell am instrument ... -e class com.faceswaplocal.app.inference.StageETwoPassCoordinatorInstrumentedTest` | OK — 1 test, 1 006,460 с |
| Регрессия E1 (UI качества) | `adb shell am instrument ... -e class com.faceswaplocal.app.ui.StageEFaceQualityUiInstrumentedTest` | OK — 3 tests, 75,006 с |
| Регрессия D (UI назначений) | `adb shell am instrument ... -e class com.faceswaplocal.app.ui.StageDComposeUiInstrumentedTest` | OK — 1 test, 48,056 с |
| Manifest | merged manifest debug и release | OK — `INTERNET`/`ACCESS_NETWORK_STATE`: 0; `WRITE_EXTERNAL_STORAGE`/`READ_MEDIA_IMAGES`: 0 |
| Устройство | `adb install -r app-debug.apk` + ручное сохранение на AVD API 35 | OK — см. ниже |

Финальная обязательная команда после последней кодовой правки:

```powershell
.\gradlew.bat test lint assembleDebug assembleDebugAndroidTest --no-daemon --console=plain
```

Результат: `BUILD SUCCESSFUL` за 5 мин 20 с. Все инструментальные прогоны выше выполнены
на APK, собранном этой командой, на `emulator-5554` (API 35 x86_64, `airplane_mode_on=1`).

Ручной сценарий на устройстве (синтетический debug-harness, без личных фотографий):

- нажата кнопка «Сохранить в галерею»;
- в UI появились миниатюра, имя `FaceSwapLocal_20260803_113153.jpg`, «Альбом
  FaceSwapLocal · 64×64» и кнопка «Открыть»;
- `adb shell ls /sdcard/Pictures/FaceSwapLocal/` — файл на месте, 1 748 байт;
- `content query --uri content://media/external/images/media` — `is_pending=0`,
  `width=64`, `height=64`;
- `run-as com.faceswaplocal.app ls cache/export` — каталог пуст, staging-файл удалён;
- файл выгружен и проверен: сегмент `Exif` присутствует, значения `FaceSwapLocal` и
  `Edited image` записаны.

Скриншоты API 35 на синтетическом debug harness:

- `docs/reports/img/STAGE_E2_PROGRESS_API35.png` — этап, «лицо 2 из 3», шкала и
  активная кнопка «Отменить» при заблокированной кнопке запуска;
- `docs/reports/img/STAGE_E2_EXPORT_API35.png` — выбор JPEG/PNG, качество 95,
  включённый по умолчанию водяной знак и кнопка сохранения;
- `docs/reports/img/STAGE_E2_SAVED_API35.png` — миниатюра, имя файла, альбом и
  системная кнопка «Открыть» после реального сохранения.

## 3. Изменённые файлы

Новые:

- `app/src/main/java/com/faceswaplocal/app/domain/ExportSettings.kt` — формат, качество
  и водяной знак без зависимости от Android-энкодера.
- `app/src/main/java/com/faceswaplocal/app/domain/ProcessingProgress.kt` — этапы и
  детерминированный расчёт доли выполнения.
- `app/src/main/java/com/faceswaplocal/app/data/ExportNaming.kt` — имя файла и альбом.
- `app/src/main/java/com/faceswaplocal/app/data/ResultExporter.kt` — staging, назначение
  `MediaStore`/SAF, метаданные, публикация и удаление незавершённых данных.
- `app/src/main/java/com/faceswaplocal/app/data/ResultWatermark.kt` — видимый знак в
  копии, уходящей в файл.
- `app/src/main/java/com/faceswaplocal/app/data/PngEditMarker.kt` — вставка `tEXt` в PNG.
- `app/src/test/java/com/faceswaplocal/app/data/ExportNamingTest.kt`,
  `PngEditMarkerTest.kt` — формат имени и корректность PNG-чанка.
- `app/src/test/java/com/faceswaplocal/app/domain/ProcessingProgressTest.kt`,
  `ExportSettingsTest.kt` — расчёт прогресса и маппинг настроек.
- `app/src/test/java/com/faceswaplocal/app/ui/ExportSettingsSavedStateTest.kt` —
  сохранение настроек экспорта примитивами.
- `app/src/androidTest/java/com/faceswaplocal/app/data/StageE2ExportInstrumentedTest.kt` —
  успех, ошибка записи, отмена, PNG, водяной знак, путь API 28 и уборка.
- `app/src/androidTest/java/com/faceswaplocal/app/inference/StageE2CancellationInstrumentedTest.kt` —
  реальная отмена multi-face прогона.
- `app/src/androidTest/java/com/faceswaplocal/app/ui/StageE2ProgressUiInstrumentedTest.kt` —
  прогресс, блокировки, поворот и настройки экспорта.

Изменённые:

- `app/src/main/java/com/faceswaplocal/app/inference/OnnxMultiPhotoFaceSwapPipeline.kt` —
  callback прогресса по этапам и лицам; логика пайплайна не менялась.
- `app/src/main/java/com/faceswaplocal/app/ui/FaceSwapViewModel.kt` — фаза
  `CANCELLING`, `photoSwapRunId`, состояние и настройки экспорта, уборка временных
  данных.
- `app/src/main/java/com/faceswaplocal/app/ui/FaceSwapApp.kt` — экран прогресса с
  отменой, карточка экспорта, SAF-запрос назначения для API 28.
- `app/src/debug/java/com/faceswaplocal/app/ui/StageDUiTestActivity.kt` — новые
  callbacks и intent-extra для скриншотов состояния обработки.
- `PRIVACY.md`, `README.md`, `docs/KNOWN_LIMITATIONS.md` — состояние после контрольной
  точки.

## 4. Benchmark и parity

Этой контрольной точкой parity не предусмотрен: пиксели результата не меняются.
Регрессионные прогоны E1 (`StageETwoPassCoordinatorInstrumentedTest`) подтверждают,
что добавление callback прогресса не изменило пайплайн. Benchmark экспорта на
физическом ARM reference device не выполнялся — устройство недоступно (§12).

## 5. Отклонения от ТЗ

- **`minSdk` оставлен равным 28.** Отложенная запись `IS_PENDING` появилась в API 29,
  а на API 28 вставка в общую коллекцию `MediaStore` потребовала бы
  `WRITE_EXTERNAL_STORAGE`, то есть полного доступа к хранилищу, прямо запрещённого
  §5.1. Поднимать `minSdk` до 29 нельзя: §18 требует установки на API 28+. Поэтому
  выбран запасной путь: на API 28 экспорт возвращает `NeedsDestination`, UI открывает
  системный `ACTION_CREATE_DOCUMENT` с предложенным именем, и запись идёт в выбранный
  пользователем документ. Разрешения при этом не требуются, а при ошибке документ
  удаляется через `DocumentsContract.deleteDocument`. Путь покрыт инструментальным
  тестом с `sdkInt = 28`, но на реальном устройстве API 28 не проверялся — доступен
  только AVD API 35.
- **Используется платформенный `android.media.ExifInterface`, а не
  `androidx.exifinterface`.** Lint выдаёт на это предупреждение (33-е из 33). Причина:
  запись отметки нужна только для JPEG, который платформенный класс поддерживает с
  API 24, и это не требует новой сторонней зависимости в полностью локальном
  приложении. Поведение подтверждено инструментальным тестом на реальном устройстве, а
  не документацией. Для PNG отметка реализована самостоятельно, без зависимости.
- **Отмена не мгновенная.** Останов происходит на ближайшей границе `ensureActive`
  координатора: начатый inference одного лица доводится до конца. Это соответствует
  формулировке FR-PHOTO-07 «на ближайшей безопасной границе»; прерывание внутри
  нативного вызова ONNX Runtime невозможно.

## 6. Известные проблемы и ограничения

- `BitmapLoader` ограничивает длинную сторону цели 2 560 px, поэтому для снимка больше
  этого размера экспортируется обработанный размер, а не исходный размер файла из
  галереи. Полноразмерная обработка 12 МП входит в отложенные до физического reference
  device пункты DoD E2 (§12, §15).
- Экран во время обработки не удерживается от засыпания (§9.4) — относится к
  оставшейся части E2.
- Пункт ручной приёмки §14.4 п. 8 (пять циклов без роста heap) и п. 12 (thermal) не
  выполнялись: они явно отложены до физического ARM reference device.
- Путь SAF для API 28 проверен только эмулированно (`sdkInt = 28` на AVD API 35).
- Экспорт предлагает только JPEG и PNG; других контейнеров нет, поэтому вопрос
  «формат не поддерживает отметку о редактировании» на практике не возникает.

## 7. Блокеры

- Нет.

## 8. Следующий шаг

- Контрольная точка 2 этапа E2: пресеты качества «Быстро/Баланс/Максимум» с реальной
  разницей в пайплайне, сила сохранения идентичности, ручная растушёвка границы маски
  и переключатель цветокоррекции авто/выключено (FR-PHOTO-05), затем профилирование
  памяти.
