# Отчёт: Этап E2, контрольная точка 6 — снятие полнокадрового warpedMask
Дата: 17.08.2026, ветка/коммит: `main`

Полнокадровая маска альфы снята с продакшена: пиксельная стоимость упала с 28 до 24 байт.
Бюджет на AVD вырос примерно на 17%, но **16 МП по-прежнему не проходит без уменьшения** —
разрыв оказался кратно больше экономии. Заодно закрыт ещё один долг по тестам.

## 1. Сделано

### 1.1 `warpedMask` стал опциональным

`FaceCompositor.pasteBack` выделял `FloatArray` на весь кадр при каждой вставке. В
продакшене его никто не читал: единственные упоминания — запись в самом компоновщике и
`paste.warpedMask.fill(0f)` в `OnnxFaceEnhancerPipeline`. На 16 МП это 64 МБ впустую,
четверть всей пиксельной стоимости.

Добавлен параметр `collectWarpedMask` (по умолчанию `false`) в `pasteBack` и `composite`;
поле в `PasteBackResult` и `FaceCompositeResult` стало `FloatArray?`. Продакшен не
выделяет ничего.

Полностью удалять поле я не стал намеренно: на нём держатся юнит-тесты, проверяющие, что
`blendConstraintMask` только уменьшает альфу и что защита обнуляет её — ровно те
инварианты, вокруг которых была контрольная точка 3. Тесты теперь запрашивают маску явно,
и покрытие сохранено. Новый тест `the warped mask is not allocated unless requested`
закрепляет и отсутствие выделения по умолчанию, и то, что запрос маски не меняет ни одного
пикселя композита.

### 1.2 Бюджет: перепись буферов уменьшилась с четырёх до трёх

`ImageMemoryBudget.JAVA_BYTES_PER_PIXEL` снижен с `4 * 4` до `3 * 4`; перепись в KDoc
переписана, вычеркнутый буфер описан явно, нумерация native-части поправлена. Итоговая
пиксельная стоимость — 24 байта вместо 28.

### 1.3 Долг: два теста были сломаны точкой 3 и не запускались

`StageE2FullResolutionInstrumentedTest` падал с `expected:<4000> but was:<1975>`. Причина
не в этой точке: его вспомогательный `budgetOf` собирает «устройство, которое может
позволить себе N пикселей», но не добавляет резерв сессий, введённый в контрольной точке 3.
С тех пор помощник недодавал около 599 МиБ, и 12-МП фикстура молча уменьшалась. Тот же
дефект был в моём собственном `RealPhotoOrientationInstrumentedTest`.

Не поймал я это потому, что мои прогоны регрессий перебирали классы пакета `inference`, а
оба теста лежат в `data`. Оба помощника исправлены, оба класса зелёные.

## 2. Проверки

| Проверка | Команда | Результат |
| --- | --- | --- |
| Unit tests + lint + сборка | `.\gradlew.bat test lint assembleDebug assembleDebugAndroidTest --no-daemon --console=plain` | OK — 100 тестов в debug и 100 в release, 0 failures; lint 0 errors, 33 warnings |
| Raw parity против FaceFusion | `am instrument ... FaceFusionParityInstrumentedTest` | OK — 1 test |
| Финальный кадр против FaceFusion | `am instrument ... FaceFusionFinalFrameParityInstrumentedTest` | OK — 1 test |
| Parser-маска свапа | `am instrument ... StageESwapParserMaskInstrumentedTest` | OK — 2 tests |
| Цветовая диагностика | `am instrument ... StageEColorDiagnosticsInstrumentedTest` | OK — 1 test |
| Плотная фикстура, шов | `am instrument ... StageEDenseUnassignedFaceInstrumentedTest` | OK — 2 tests |
| Геометрия лица | `am instrument ... FaceQualityGeometryInstrumentedTest` | OK — 1 test |
| Качество лица | `am instrument ... FaceQualityParityInstrumentedTest` | OK — 1 test |
| Групповой сценарий этапа D | `am instrument ... StageDMultiFaceInstrumentedTest` | OK — 1 test |
| Барьер двух проходов | `am instrument ... StageETwoPassCoordinatorInstrumentedTest` | OK — 1 test |
| Полное разрешение и бюджет | `am instrument ... data.StageE2FullResolutionInstrumentedTest` | OK — 4 tests |
| EXIF-ориентация | `am instrument ... data.RealPhotoOrientationInstrumentedTest` | OK — 1 test |
| Экспорт | `am instrument ... data.StageE2ExportInstrumentedTest` | OK — 7 tests |
| 12 МП сквозной прогон | `am instrument ... StageE2FullResolutionPipelineInstrumentedTest` | OK — 1 test |
| Отмена свапа | `am instrument ... StageE2CancellationInstrumentedTest` | OK — 1 test |
| Отмена восстановления | `am instrument ... StageE2RestorationCancellationInstrumentedTest` | OK — 1 test |
| Гигиена эмбеддингов | `am instrument ... StageEEmbeddingHygieneInstrumentedTest` | OK — 2 tests |

Оба parity-теста против FaceFusion проходят без изменения порогов — снятие маски не
сдвинуло ни одного пикселя, чего и следовало ожидать: поле только писали.

## 3. Изменённые файлы

- `app/src/main/java/com/faceswaplocal/app/inference/FaceCompositor.kt` — `collectWarpedMask`,
  nullable-поля, выделение только по запросу.
- `app/src/main/java/com/faceswaplocal/app/inference/OnnxFaceEnhancerPipeline.kt` — очистка
  через `?.`.
- `app/src/main/java/com/faceswaplocal/app/data/ImageMemoryBudget.kt` — три буфера вместо
  четырёх, перепись в KDoc.
- `app/src/test/.../FaceCompositorTest.kt` — явный запрос маски там, где она проверяется,
  плюс тест на отсутствие выделения по умолчанию.
- `app/src/androidTest/.../StageE2FullResolutionInstrumentedTest.kt`,
  `.../RealPhotoOrientationInstrumentedTest.kt` — резерв сессий в помощниках бюджета.
- `app/src/androidTest/.../StageESwapParserMaskInstrumentedTest.kt` — очистка через `?.`.
- `docs/reports/STAGE_E2_CHECKPOINT_6_REPORT.md` — этот отчёт.

## 4. Измерения

### 4.1 Бюджет вырос, но 16 МП не достаёт

A/B в одной сессии на одном AVD, по логу `StageE2Budget`:

| Стоимость пикселя | `maxTargetPixels` |
| --- | --- |
| 28 байт (было) | 9 532 342 |
| 24 байта (стало) | 11 328 853 … 12 577 109 |

Четыре подряд идущих замера после снятия дали 11 328 853, 11 667 200, 11 706 794 и
11 747 840; 12 577 109 получен сразу после холодной загрузки и является выбросом вверх.
Разброс — колебания `availMem` эмулятора, а не свойство правки. Детерминированная часть
выигрыша считается точно: при равной доступной памяти бюджет растёт ровно в 28/24 = 1,167
раза, то есть на 16,7%.

**16 МП по-прежнему уменьшается.** Кадру 4895×3268 нужно 15 996 860 пикселей, бюджет даёт
около 11,7 млн, то есть снимок декодируется примерно как 4340×2897 (12,6 млн) вместо
3694×2466 (9,1 млн) до правки. Ожидание «не хватало примерно этого порядка» не
подтвердилось: снятие 4 байт из 28 даёт +16,7%, а чтобы дойти с 9,1 до 16,0 МП, нужно
+76%. Экономия реальная, но кратно меньше разрыва.

Побочное наблюдение: порог 12 МП теперь проходится не всегда. `fitsTwelveMegapixels` был
`true` при 12,58 млн и `false` при 11,60 млн — то есть 12 МП на этом эмуляторе балансирует
на грани и зависит от того, сколько памяти занято системой в момент запуска.

### 4.2 Что осталось бы сделать ради полного 16 МП

Оставшиеся 24 байта — это три полнокадровых буфера Java (накопленный композит, копия базы
внутри `pasteBack`, копия `readPixels`) и три native (декодированная цель, предыдущий
рабочий bitmap, новый результат). Убрать ещё один без смены архитектуры не выйдет: каждый
живёт одновременно с остальными по построению текущего пайплайна. Полное разрешение
требует тайлового компоновщика — обработки кадра полосами, — а это отдельная работа, не
константа.

## 5. Отклонения от ТЗ

- Цель «16 МП доходит до экспорта без уменьшения» **не достигнута** (§4.1). Правка сделана
  и даёт измеренный выигрыш, но его недостаточно.
- Поле `warpedMask` не удалено полностью, а переведено в опциональное — иначе терялись
  юнит-тесты инвариантов защиты и ограничения альфы (§1.1).

## 6. Известные проблемы и ограничения

- 16 МП на двухгигабайтном AVD по-прежнему уменьшается; в `docs/KNOWN_LIMITATIONS.md` это
  уже описано как свойство стенда. Цифру порога там стоит обновить при следующем касании:
  было около 9,1 МП, стало около 11,7 МП.
- 12 МП проходит не при каждом запуске (§4.1).
- Мои прогоны регрессий до этой точки не покрывали пакеты `data` и `ui`. Пакет `data`
  теперь прогнан целиком; `ui` (пять Compose-классов) в этой точке не запускался — правка
  их не касается, но полным набором это назвать нельзя.

## 7. Блокеры

Нет.

## 8. Следующий шаг

Контрольная точка 7 — пресеты качества и закрытие E2: «Быстро» без восстановления,
«Баланс» с GFPGAN 0,8, «Максимум» с GFPGAN и parser-маской блендинга, с показом ожидаемого
времени по фактическим замерам.
