# Отчёт: Этап A — Воспроизводимая базовая сборка
Дата: 19.07.2026, ветка/коммит: `main` / `6835ff1ad00ace193dc342e293da97d0b2d140cc`

## 1. Сделано
- Репозиторий `https://github.com/Nodensss/faceswaps.git` клонирован в `C:\Users\ozr\Documents\facees`; исходный отчёт этапа A перенесён в клонированный репозиторий и затем актуализирован.
- `AGENTS.md`, `README.md` и исходная версия ТЗ прочитаны; корневой `TECHNICAL_SPEC.md` заменён полным текстом версии 2.0 от 18.07.2026, предоставленным пользователем. SHA-256 файла: `B3DCB00597A33889947D44718D3EA9E058D16DA3F16A2542FCA6C1A7335482FE`.
- Проверены JDK Temurin 17.0.19, Android SDK Platform 35, Build Tools 35.0.0, platform-tools 37.0.0, Emulator 36.5.11 и Gradle Wrapper 8.9.
- Gradle-проект успешно сконфигурирован через Wrapper; исправлены ошибочные явные импорты Compose, блокировавшие компиляцию.
- В merged manifest удалены транзитивно добавленные ML Kit разрешения `INTERNET`, `ACCESS_NETWORK_STATE`, сетевой backend и фоновые планировщики DataTransport; необходимые ML Kit runtime-классы сохранены, детекция лиц работает локально.
- Правило начального сопоставления приведено к FR-PHOTO-04: первый источник автоматически назначается только первому целевому лицу; явное назначение ранее неназначенной цели корректно добавляется.
- Unit-тесты сопоставления обновлены; в CI добавлен обязательный запуск `lint`.
- Debug APK установлен на эмулятор Android 15 / API 35. В авиарежиме на синтетическом изображении выполнен сценарий: выбор источника и цели через системный Photo Picker → обнаружение двух лиц на каждом изображении → автоматическое назначение источника 1 цели 1 → ручное назначение источника 2 цели 2.
- Доказательства ручной проверки: [обнаруженные лица](img/stage_a_faces_detected.png) и [назначение источников](img/stage_a_mapping_assigned.png). В изображениях только синтетические вымышленные люди; личные фотографии не использовались.

### Definition of Done этапа A
- Gradle Sync/конфигурация проекта проходит — **выполнено**.
- `test`, `lint` и `assembleDebug` проходят — **выполнено**.
- Приложение запускается на устройстве API 35 — **выполнено**.
- Выбор фото, multi-face detection и mapping UI работают на устройстве — **выполнено**.
- Итог — **этап A закрыт**.

## 2. Проверки
| Проверка | Команда | Результат |
| --- | --- | --- |
| Gradle Sync / конфигурация | `.\gradlew.bat help --stacktrace` | OK — `BUILD SUCCESSFUL` |
| Unit tests | `.\gradlew.bat test --console=plain` | OK — 4 теста для debug + 4 для release, 0 failures/errors/skips |
| Lint | `.\gradlew.bat lint --console=plain` | OK — 0 ошибок, 23 предупреждения |
| Сборка | `.\gradlew.bat assembleDebug --console=plain` | OK — `BUILD SUCCESSFUL`; APK 75 289 944 байта |
| Manifest APK | `aapt.exe dump permissions app-debug.apk`; поиск `INTERNET`, `ACCESS_NETWORK_STATE` и DataTransport в merged manifest | OK — сетевых разрешений, backend и планировщиков DataTransport нет |
| Устройство | `adb devices`; `adb install -r app/build/outputs/apk/debug/app-debug.apk`; ручной сценарий | OK — `emulator-5554`, Android 15 / API 35, установка `Success`, сценарий выполнен в авиарежиме |

Артефакт: `app/build/outputs/apk/debug/app-debug.apk`, SHA-256 `23F34B187A5556F386B7AE0D9889BD36199E54EE600422E24F44E5393191E97E`.

## 3. Изменённые файлы
- `TECHNICAL_SPEC.md` — точная предоставленная версия ТЗ 2.0.
- `.github/workflows/android.yml` — добавлен запуск lint в базовую CI-проверку.
- `app/src/main/AndroidManifest.xml` — удаление запрещённых сетевых разрешений и фоновых компонентов из merged manifest.
- `app/src/main/java/com/faceswaplocal/app/domain/FaceModels.kt` — корректные начальные и явные назначения источников.
- `app/src/main/java/com/faceswaplocal/app/ui/FaceSwapApp.kt` — исправлены импорты Compose, блокировавшие сборку.
- `app/src/test/java/com/faceswaplocal/app/domain/FaceAssignmentPlannerTest.kt` — тесты FR-PHOTO-04 и назначения ранее неназначенной цели.
- `docs/reports/img/stage_a_faces_detected.png` — доказательство multi-face detection.
- `docs/reports/img/stage_a_mapping_assigned.png` — доказательство явного mapping.
- `docs/reports/STAGE_A_REPORT.md` — актуальный отчёт этапа A.

## 4. Benchmark и parity (если предусмотрены этапом)
- Benchmark inference и parity-тест для этапа A не предусмотрены.
- Устройство ручной проверки: AVD `QuizMaster_API35`, модель `Android SDK built for x86_64`, Android 15 / API 35. Нейросетевой backend на этапе A не подключался.

## 5. Отклонения от ТЗ
- Android Studio не установлен в стандартных каталогах и не доступен в `PATH`; Gradle Sync выполнен эквивалентной полной конфигурацией проекта через Gradle Wrapper: `.\gradlew.bat help --stacktrace`.
- Захват окна эмулятора через Computer Use завершился ошибкой `SetIsBorderRequired failed: Интерфейс не поддерживается (0x80004002)`; ручная проверка выполнена через `adb` (`uiautomator`, ввод и снимки экрана). Функциональный объём проверки не сокращён.

## 6. Известные проблемы и ограничения
- Нейросетевая замена лиц ещё не реализована; это ожидаемая граница этапа A, подключение моделей относится к этапу B.
- Lint завершён без ошибок, но содержит 23 предупреждения: 18 `GradleDependency`, 3 `AndroidGradlePluginVersion`, 1 `DataExtractionRules`, 1 `OldTargetApi`. Обновления версий не выполнялись вне объёма этапа A.
- Instrumentation-тестов текущего сценария в проекте пока нет; на этапе A сценарий подтверждён вручную на эмуляторе.

## 7. Блокеры
- Нет.

## 8. Следующий шаг
- После подтверждения пользователя выполнить этап B: документировать выбранные модели из shortlist, реализовать безопасный локальный импорт весов, подключить ONNX Runtime и выполнить parity-тест геометрии и сырого выхода swapper.
