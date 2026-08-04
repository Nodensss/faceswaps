# Отчёт: Этап E2, контрольная точка 2 — полное разрешение и удержание экрана
Дата: 04.08.2026, ветка/коммит: `main` / `<hash>`

Этап E2 целиком не закрыт. Эта контрольная точка закрывает два пункта: адаптивный
предел разрешения вместо константы 2 560 px (§5.2, §12) и запрет засыпания экрана во
время обработки (§9.4). Пресеты качества, сила сохранения идентичности, ручная
растушёвка и переключатель цветокоррекции отложены по прямому указанию пользователя.

## 1. Сделано

- **Константа 2 560 px убрана для целевого фото.**
  `app/src/main/java/com/faceswaplocal/app/data/ImageMemoryBudget.kt` считает предел из
  фактической свободной Java-кучи (`Runtime.maxMemory()` минус занятое) и доступной
  системной памяти (`ActivityManager.MemoryInfo.availMem` минус `threshold`), берёт
  меньшее из двух и вдвое ужимает его, пока система сообщает `lowMemory`. Множители
  выведены из переписи полнокадровых буферов пайплайна, а не подобраны: 16 байт на
  пиксель в Java-куче (накопленный композит, `basePixels.copyOf()`, полнокадровая
  `warpedMask`, `readPixels()`-копия) и 12 байт в native-куче (декодированная цель,
  предыдущий рабочий bitmap, новый результат). Перепись записана в KDoc класса.
- **Источники сознательно остались на 2 560 px по длинной стороне.** Из источника
  берётся только выровненный кроп 112×112, их может быть восемь одновременно, и все
  parity-фикстуры этапов B–E1 измерялись именно через этот предел. Поднимать его —
  значит потратить память и поставить под вопрос уже принятый parity.
- **`BitmapLoader` разделён на `loadTarget` и `loadSource`** и возвращает `DecodedImage`
  с исходными размерами файла, поэтому UI знает, совпал ли обработанный размер с
  оригиналом.
- **Бюджет — жёсткий потолок, а не приблизительный.** Обе стороны округляются вниз:
  независимое округление к ближайшему давало 4000×3000 → 1633×1225 = 2 000 425 px при
  бюджете 2 000 000. Для источников, где ограничение задано на сторону, а не на
  произведение, длинная сторона попадает в предел точно.
- **Честное сообщение при нехватке памяти.** Если бюджет заставил уменьшить цель,
  карточка экспорта прямо пишет исходный и обработанный размеры перед сохранением
  (`export-downscale-notice`), а не молча отдаёт файл меньшего размера.
- **`android:largeHeap="true"` в манифесте.** Это не косметика: `dalvik.vm.heapgrowthlimit`
  на устройстве класса 2 ГБ равен `192m`, и адаптивный бюджет при таком потолке
  ограничил бы любое фото примерно 5 МП, что противоречит §5.2. С `largeHeap`
  `Runtime.maxMemory()` = 512 МиБ и бюджет на этом AVD равен 20 039 691 px. Бюджет
  по-прежнему читает фактический потолок, то есть ограничение не отменено, а поднято.
- **Экран не гаснет во время обработки (§9.4).** `KeepScreenOnWhile` в
  `app/src/main/java/com/faceswaplocal/app/ui/FaceSwapApp.kt` держит флаг на `View`
  композиции ровно пока `isProcessing`, и снимает его в `onDispose`, поэтому ошибка и
  отмена освобождают экран так же, как успешное завершение.

## 2. Проверки

| Проверка | Команда | Результат |
| --- | --- | --- |
| Unit tests | `.\gradlew.bat test --no-daemon --console=plain` | OK — 94 теста в debug и 94 в release, 0 failures/errors/skipped |
| Lint | `.\gradlew.bat lint --no-daemon --console=plain` | OK — 0 errors, 33 warnings |
| Сборка | `.\gradlew.bat assembleDebug --no-daemon --console=plain` | OK — `app-debug.apk`, 180 130 651 байт |
| 12 МП сквозной прогон | `adb shell am instrument ... -e class com.faceswaplocal.app.inference.StageE2FullResolutionPipelineInstrumentedTest` | OK — 1 test, 104,307 с |
| Бюджет и декодирование | `adb shell am instrument ... -e class com.faceswaplocal.app.data.StageE2FullResolutionInstrumentedTest` | OK — 4 tests, 2,154 с |
| Удержание экрана | `adb shell am instrument ... -e class com.faceswaplocal.app.ui.StageE2KeepScreenOnInstrumentedTest` | OK — 2 tests, 29,424 с |
| 12 МП в Compose | `adb shell am instrument ... -e class com.faceswaplocal.app.ui.StageE2LargeBitmapUiInstrumentedTest` | OK — 2 tests, 38,026 с |
| Регрессия E2 (экспорт) | `adb shell am instrument ... -e class com.faceswaplocal.app.data.StageE2ExportInstrumentedTest` | OK — 7 tests, 2,843 с |
| Регрессия E2 (прогресс) | `adb shell am instrument ... -e class com.faceswaplocal.app.ui.StageE2ProgressUiInstrumentedTest` | OK — 5 tests, 79,384 с |
| Регрессия E1 (два прохода) | `adb shell am instrument ... -e class com.faceswaplocal.app.inference.StageETwoPassCoordinatorInstrumentedTest` | OK — 1 test, 819,716 с |
| Устройство | `adb install -r app-debug.apk` на AVD API 35, `airplane_mode_on=1` | OK |

Финальная обязательная команда после последней кодовой правки:

```powershell
.\gradlew.bat test lint assembleDebug assembleDebugAndroidTest --no-daemon --console=plain
```

Результат: `BUILD SUCCESSFUL` за 3 мин 8 с. Прогон 12 МП, Compose-тест 12 МП и
`StageE2LargeBitmapUiInstrumentedTest` выполнены на APK, собранном этой командой;
остальные инструментальные классы прогонялись на предыдущей сборке того же кода.

Ошибка процесса в первом прогоне регрессии E1 была моей: я запустил её параллельно с
другими инструментальными классами, и `am instrument` force-stop'нул общий процесс.
`dumpsys activity exit-info` показал `reason=10 (USER REQUESTED) subreason=21 (FORCE
STOP) ... due to start instr`, то есть ни OOM, ни падения не было. Повторный
одиночный прогон дал `OK (1 test)`.

## 3. Изменённые файлы

Новые:

- `app/src/main/java/com/faceswaplocal/app/data/ImageMemoryBudget.kt` — адаптивный
  предел, перепись буферов и `RuntimeMemory`-шов для тестов.
- `app/src/test/java/com/faceswaplocal/app/data/ImageMemoryBudgetTest.kt`,
  `DecodeSizePolicyTest.kt` — формула бюджета и арифметика размеров.
- `app/src/androidTest/java/com/faceswaplocal/app/data/StageE2FullResolutionInstrumentedTest.kt`
  — декодирование 12 МП, узкий бюджет, предел источников, фактический бюджет устройства.
- `app/src/androidTest/java/com/faceswaplocal/app/inference/StageE2FullResolutionPipelineInstrumentedTest.kt`
  — сквозной прогон 12 МП с замером памяти и проверкой paste ROI.
- `app/src/androidTest/java/com/faceswaplocal/app/ui/StageE2KeepScreenOnInstrumentedTest.kt`
  — удержание экрана только на время обработки.
- `app/src/androidTest/java/com/faceswaplocal/app/ui/StageE2LargeBitmapUiInstrumentedTest.kt`
  — 12 МП в viewport «До/После» и предупреждение об уменьшении.

Изменённые:

- `app/src/main/java/com/faceswaplocal/app/data/BitmapLoader.kt` — `loadTarget`/
  `loadSource`, `DecodedImage` и вынесенная `DecodeSizePolicy`.
- `app/src/main/AndroidManifest.xml` — `android:largeHeap="true"` с обоснованием.
- `app/src/main/java/com/faceswaplocal/app/ui/FaceSwapViewModel.kt` — размер исходного
  файла в состоянии и `exportIsDownscaled`.
- `app/src/main/java/com/faceswaplocal/app/ui/FaceSwapApp.kt` — `KeepScreenOnWhile` и
  предупреждение об уменьшенном экспорте.
- `README.md`, `docs/KNOWN_LIMITATIONS.md`, `docs/BENCHMARKS.md` — состояние и замеры.

## 4. Benchmark и parity

Полные числа — `docs/BENCHMARKS.md`, раздел «E2, контрольная точка 2». Машинный
артефакт прогона: `files/stage_e2_full_resolution.json` в приватном хранилище
приложения (`adb shell run-as com.faceswaplocal.app cat files/stage_e2_full_resolution.json`).

Экспорт выполнен настройками по умолчанию (JPEG q95, водяной знак включён): знак
копирует весь bitmap результата, поэтому измерение без него занизило бы реальный пик.

| Величина | Значение |
| --- | ---: |
| Размер цели | 4000×3000 (12 000 000 px) |
| Бюджет в момент декодирования | 20 039 538 px |
| Декодировано в полном разрешении | да |
| Время координатора (свап + GFPGAN 0.8, одно лицо) | 87 346 ms |
| Peak Java heap | 169 424 624 B (161,6 МиБ) |
| Peak native heap allocated | 1 309 729 840 B (1 249,1 МиБ) |
| Peak total PSS | 1 357 218 816 B (1 294,3 МиБ) |
| `Runtime.maxMemory()` | 536 870 912 B (512 МиБ) |
| Изменено пикселей вне union(swap, enhance) ROI | 0 из 12 000 000 |
| Размер экспортированного файла | 4000×3000 |

**2-гигабайтный AVD вытянул 12 МП.** Пиковый суммарный PSS 1 294,3 МиБ укладывается в
ориентир §12 (~1,5 ГБ) и занимает около 86% от него. Пиковая Java-куча 161,6 МиБ
измерена против потолка 512 МиБ; при штатных 192 МиБ такого запаса не было бы — это и
есть измеренное обоснование `largeHeap`.

Тот же сценарий с выключенным водяным знаком дал peak native 1 191,6 МиБ и peak PSS
1 244,1 МиБ, то есть включённый по умолчанию знак стоит около 57,5 МиБ native — ровно
один полный кадр, потому что он рисуется в копии и не трогает bitmap результата.

Перепись буферов, на которой построен бюджет, согласуется с замером: модель
предсказывает 16 байт на пиксель в Java-куче, то есть 192 МиБ на 12 МП, фактический
пик — 161,6 МиБ вместе со всем остальным содержимым кучи. Модель консервативна и не
занижает потребление.

Parity этой контрольной точкой не предусмотрен: numeric-пути свапа и восстановления не
менялись, а все parity-фикстуры меньше 2 560 px, поэтому проходят через новый бюджет
без изменения размера. Регрессия E1 подтверждена отдельным прогоном.

## 5. Отклонения от ТЗ

- **Добавлен `android:largeHeap="true"`.** ТЗ его не требует и не запрещает; §12
  наоборот закладывает пик до ~1,5 ГБ. Без него требование §5.2 «для фото сохранять
  исходное разрешение результата» физически недостижимо на устройстве класса 2 ГБ:
  бюджет упирался бы в 192 МиБ кучи и резал бы цель примерно до 5 МП. Решение
  измерено, а не предположено, и записано в `docs/BENCHMARKS.md`.
- **Предел для источников оставлен константой 2 560 px.** Формально это тоже
  константа, но она относится к изображению, из которого берётся только выровненный
  кроп 112×112, и совпадает с условиями, в которых снят весь parity этапов B–E1.
  Менять её означало бы пересчитывать parity ради нулевого выигрыша в качестве.
- **Полнокадровая `warpedMask` в `FaceCompositor` не оптимизирована.** На 12 МП это
  48 МиБ Java-кучи на вызов, и в production её никто не читает — только тесты. Замер
  показал, что запас есть, поэтому компоновщик, покрытый parity-тестами E1, не
  трогался. Это зафиксировано как кандидат на снижение пика, а не как сделанная работа.

## 6. Известные проблемы и ограничения

- Запас по памяти на 2 ГБ AVD небольшой: 1 294 МиБ PSS против ориентира ~1,5 ГБ. На
  телефоне с меньшим лимитом кучи адаптивный бюджет сам уменьшит цель; поведение
  покрыто тестом узкого бюджета и предупреждением в UI, но на реальном устройстве с
  малой памятью не проверялось.
- Ручной сценарий «выбрать 12 МП фото системным picker и обработать в UI» не
  выполнялся: проверен программный путь целиком (декодирование → координатор →
  экспорт) и отдельный Compose-тест на отрисовку 12 МП в viewport «До/После».
- Пять повторных циклов без устойчивого роста heap и поведение при thermal throttling
  не проверялись — §15 откладывает эти пункты DoD E2 до физического ARM reference
  device.
- Пресеты качества, сила сохранения идентичности, ручная растушёвка границы маски и
  переключатель цветокоррекции авто/выключено не начинались.

## 7. Блокеры

- Нет.

## 8. Следующий шаг

- Контрольная точка 3 этапа E2: пресеты качества «Быстро/Баланс/Максимум» с реальной
  разницей в пайплайне, сила сохранения идентичности, ручная растушёвка границы маски
  и переключатель цветокоррекции авто/выключено (FR-PHOTO-05).
