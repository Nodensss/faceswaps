# Аудит этапов A–D
Дата: 23.07.2026. Аудитор: Claude (независимая проверка по коду, не по тексту отчётов).
Коммит на момент аудита: `main` / `64dd3da73a9afd9da29940b2ff31b3b9bc4cb113`.

Метод: утверждения из отчётов A–D сверялись с исходным кодом, конфигурацией сборки и
фактическими артефактами. Код не изменялся. Формулировки прямые: где не сходится —
сказано прямо.

---

## 0. Бэкап (выполнено)

- `git remote -v`: единственный remote `origin` = `https://github.com/Nodensss/faceswaps.git`.
- До пуша `main` опережал `origin/main` на 16 коммитов (`703a862..64dd3da`) — вся работа
  этапов A–D существовала только локально.
- `git ls-files "*.onnx"` — **пусто**. Веса моделей в индекс не попадали. `.gitignore`
  запрещает `*.onnx`, `*.tflite`, `*.jks`, `*.keystore`, `app/src/main/assets/models/*`.
- Выполнен `git push origin main`: `703a862..64dd3da  main -> main`.
- После пуша `git rev-parse HEAD` == `git rev-parse origin/main` == `64dd3da`.
  **Все 16 коммитов недели работы теперь на GitHub.**

---

## 1. Что подтвердилось

### 1.1 `StageDMultiFaceInstrumentedTest` — ассерты реальны
`app/src/androidTest/.../inference/StageDMultiFaceInstrumentedTest.kt`. Тест не
«просто не падает», а проверяет заявленное численно:
- ровно 4 нейролица во фикстуре (`assertEquals(4, detected.size)`);
- три **разных** файла-источника (`pair_01/02/03_source.png`) на T1/T2/T3;
- каждый из T1/T2/T3 изменён: `changedInBox(...) > 100` пикселей;
- **T4 побитово идентичен**: `assertEquals(0, changedInBox(original, finalPixels, ..., ordered[3].box))`;
- **вне объединения paste-ROI изменено 0 пикселей**: `outsideUnionChanges(...) == 0`,
  где функция честно перебирает все пиксели кадра и считает те, что вне всех ROI;
- пересечение ROI T1/T2 существует (`overlap.width>0 && overlap.height>0`) и содержит
  второй paste, а не оригинал (`changedInRoi > 20`).

Ассерты корректны и соответствуют тексту отчёта D §1.

### 1.2 `StageDComposeUiInstrumentedTest` — ассерты реальны
`app/src/androidTest/.../ui/StageDComposeUiInstrumentedTest.kt`. Проверяет:
- независимые назначения: после кликов `assign-1-1`, `assign-2-2`, `unchanged-3` →
  target-1=source-1, target-2=source-2, target-3=source-3, target-4=null;
- диалог подтверждения apply-to-all и его эффект (все 4 цели → source-1);
- инвалидацию при удалении источника (target-2 → null);
- сохранение состояния после `recreate()` (target-1=source-1, target-2=null).

Assert’ы читают реальный `viewModel.state.value.assignments`, а не только UI-узлы.
Три разных источника на трёх лицах подтверждены на уровне состояния ViewModel.

### 1.3 Кэш эмбеддингов очищается на пути отмены
`OnnxMultiPhotoFaceSwapPipeline.process` (`inference/OnnxMultiPhotoFaceSwapPipeline.kt:60-63`):
```
} finally {
    sourceEmbeddings.values.forEach { it.fill(0f) }
    sourceEmbeddings.clear()
}
```
`finally` охватывает цикл с `coroutineContext.ensureActive()`. При отмене `ensureActive`
бросает `CancellationException`, которая проходит через `finally` → зануление и очистка
map срабатывают на успехе, ошибке **и** отмене. Подтверждено.

### 1.4 `ModelStore` — полное хеширование при импорте, кэш процесс-локальный
`inference/ModelStore.kt`:
- `verifiedThisProcess` — приватное поле экземпляра `MutableSet<ModelId>`, то есть
  привязано к процессу (новый процесс — новый `ModelStore` — пустой кэш);
- **импорт всегда хеширует полностью**: `importLocked` → `copySourceToPart` →
  `ModelFileIntegrity.copyAndHash` считает SHA-256 всего потока, независимо от кэша,
  затем `verifiedThisProcess += id` уже после валидации;
- `requireVerifiedModel` вызывается непосредственно перед открытием session и требует
  статус `Ready`.

Подтверждено, что кэш не позволяет пропустить проверку импортируемого файла.

### 1.5 `SavedStateHandle` — только идентификаторы
`ui/FaceSwapViewModel.kt` + `domain/FaceModels.kt`:
- единственный сохраняемый ключ `stage_d_assignment_ids` = `ArrayList<String>` вида
  `"targetId|sourceId"` через `AssignmentStateCodec.encode`;
- никаких bitmap, эмбеддингов, URI или пиксельных данных в handle нет;
- сообщение после смерти процесса честное: «изображения нужно выбрать заново».

Подтверждено.

### 1.6 `StageDUiTestActivity` — только debug, не в release
- Файл лежит в `app/src/debug/java/.../ui/StageDUiTestActivity.kt`;
- регистрируется только в `app/src/debug/AndroidManifest.xml`;
- в собранном **release-APK** (`app-release-unsigned.apk`): в dex **нет** строки
  `StageDUiTestActivity`, в бинарном манифесте активити отсутствует.

Подтверждено прямым сканом release-артефакта.

### 1.7 Merged manifest release — без сети
- `app/src/main/AndroidManifest.xml` удаляет `INTERNET` и `ACCESS_NETWORK_STATE`
  через `tools:node="remove"` + вырезает сетевые компоненты DataTransport ML Kit;
- в merged/packaged release-манифесте (`processReleaseManifest`,
  `processReleaseManifestForPackage`) ни `INTERNET`, ни `ACCESS_NETWORK_STATE` нет;
- в бинарном манифесте release-APK обеих строк нет (проверено байтово, включая UTF-16).

Единственное uses-permission в release — авто-генерируемое
`com.faceswaplocal.app.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION` (не сетевое).
Подтверждено.

### 1.8 `make_group_fixture.py` — воспроизводится побитово
Перегенерировал фикстуру из закоммиченных синтетических портретов (seed `20260722`,
Python 3.12.10, Pillow 12.3.0):
```
352b1fb66c38ae1605e5c19476bd75e4f31efef9e2e3cc45d66f128bf3e2688c  (регенерация)
352b1fb66c38ae1605e5c19476bd75e4f31efef9e2e3cc45d66f128bf3e2688c  (закоммиченный stage_d_group_target.png)
```
SHA-256 совпал побитово. Подтверждено.

### 1.9 Личных фотографий в репозитории нет
`git ls-files` по изображениям: 48 PNG — только `docs/parity/` (синтетические входы,
эталонные и Android-выходы) и `docs/reports/img/` (скриншоты UI). Просмотрел выборку
(parity-входы, скриншоты этапов A и D) — везде синтетические студийные портреты,
консистентные с §14.3. Личных фото не обнаружено.

### 1.10 Сборка и проверки (прогнано заново)
`./gradlew test lint assembleDebug` — `BUILD SUCCESSFUL`. Отдельно форсировал
`testDebugUnitTest testReleaseUnitTest --rerun-tasks`:
- Unit tests: **42 debug + 42 release = 84 прогона, 0 failures, 0 errors** (совпадает
  с отчётом D §2, «84 прогона»);
- Lint: **0 errors, 32 warnings** (совпадает);
- assembleDebug: APK **179 944 427 байт**, SHA-256
  `57A3F0F45521F86056502DC5B6E365BAFCFAF7DDEB4B72F370ABD7198E8CC091`.

**SHA-256 APK совпал с §2 отчёта D байт-в-байт**, размер тоже. Сборка детерминирована
(повторная сборка дала тот же хеш). Дерево чистое (`git status` пустой).

Дополнительно собрал `assembleRelease` (`app-release-unsigned.apk`) для проверки
release-манифеста и dex — см. 1.6, 1.7.

---

## 2. Что не подтвердилось / требует оговорки

Ни одно ложное утверждение в отчётах не найдено. Ниже — расхождения между буквальной
формулировкой отчётов и кодом, которые не являются ошибками, но должны быть зафиксированы.

### 2.1 Инструментальные тесты я перезапустить не мог (важно)
`StageDMultiFaceInstrumentedTest` и `StageDComposeUiInstrumentedTest` требуют
эмулятора и импортированных весов (`.onnx` вне репозитория). Я подтвердил, что
**ассерты в исходниках проверяют заявленное**, но **не перезапускал** сами тесты —
веса не коммитятся, состояние `emulator-5554` в этой сессии не воспроизводилось.
Утверждение отчёта D «`OK (1 test)`» я принимаю на уровне кода, а не собственного
прогона. Для полной независимой верификации нужен эмулятор + локально импортированные
веса.

### 2.2 Multi-face instrumented test обходил продакшн-координатор — ИСПРАВЛЕНО (§5.1)
До правки `StageDMultiFaceInstrumentedTest` вызывал `OnnxPhotoFaceSwapPipeline.process` в
цикле и **сам** реконструировал порядок лиц и накопление `basePixels`. Реальный
оркестратор `OnnxMultiPhotoFaceSwapPipeline.process` этим тестом с реальным inference не
покрывался. **Исправлено 23.07.2026**: тест переписан на прямой вызов координатора и
перезапущен на эмуляторе (`OK (1 test)`). См. §5.1.

### 2.3 Удаление источника в Compose-тесте — прямой вызов, а не клик
В `StageDComposeUiInstrumentedTest` кнопка `remove-source-1` только проверяется на
существование (`fetchSemanticsNode()`), после чего инвалидация запускается прямым
вызовом `viewModel.removeSource(FaceId("source-2"))`. Это **честно раскрыто** в отчёте
D §5 (причина — нестабильный клик во вложенных скроллах API 35). Фиксирую: UI-жест
удаления через клик по факту не проверяется автотестом; проверяется только логика
инвалидации и наличие узла.

### 2.4 Зануление эмбеддинга — цитата была шире гарантии — ИСПРАВЛЕНО (§5.2)
Отчёт D §1: «В `finally` каждый `FloatArray` затирается нулями». До правки это было
верно для массивов в `sourceEmbeddings` map, но возвращаемый
`PhotoFaceSwapResult.sourceEmbedding` последнего лица уезжал в `photoSwapResult`
ViewModel незанулённым (строго в RAM, но вопреки буквальной формулировке).
**Исправлено 23.07.2026**: координатор возвращает `MultiPhotoFaceSwapResult` **без**
поля эмбеддинга, а финальная копия дополнительно зануляется в `finally`. См. §5.2.

### 2.5 Быстрый путь `ModelStore` при ревалидации — только размер
`verifyStatusLocked`: если `id in verifiedThisProcess && length == expected`, повторный
SHA-256 не считается (возвращается `Ready`). Для **импорта** это неважно (импорт всегда
хеширует полностью — см. 1.4). Но при повторном открытии session в том же процессе
подмена файла на файл того же размера в приватном `filesDir/models` не была бы поймана
повторным хешем. Модель приватна для приложения (Android sandbox), так что вне root это
недостижимо, и отчёты B §6 / D §5 это честно описывают («проверяют приватный путь и
размер»). Фиксирую как задокументированный остаточный риск, не как расхождение.

---

## 3. Что требует исправления

Блокирующих дефектов нет. По итогам аудита пользователь заказал три правки — все
выполнены и проверены (§5):

1. **(2.2) СДЕЛАНО** — тест переписан на продакшн-координатор, перезапущен на эмуляторе.
2. **(2.4) СДЕЛАНО** — эмбеддинг больше не возвращается в ViewModel, финальная копия
   зануляется.
3. **(2.3) НЕ ТРОГАЛОСЬ** — по решению пользователя оставлено как есть; зафиксировано как
   отклонение в отчёте D §5 (клик по удалению нестабилен на API 35, проверяется прямой
   вызов + unit-тест).
4. Общие ограничения B–D вне объёма A–D остаются: XNNPACK/ARM64, NNAPI, peak heap и
   thermal — без физического reference device; качество blending/restoration — этап E1.

---

## 4. Итог

Отчёты этапов A–D **достоверны**. Все ключевые проверяемые утверждения подтвердились по
коду и артефактам: ассерты обоих инструментальных тестов реальны, очистка кэша
эмбеддингов стоит в `finally` и работает на отмене, `ModelStore` всегда хеширует импорт,
`SavedStateHandle` хранит только идентификаторы, debug-активити не попадает в release,
release-манифест без сети, фикстура воспроизводится побитово, личных фото нет, а
SHA-256 debug-APK совпал с отчётом D байт-в-байт. Расхождения из §2 — это оговорки к
формулировкам и пробелы покрытия, а не ложные заявления.

После аудита выполнены три заказанные правки и выборочный parity-спот-чек (§5); все
зелёные. Бэкап выполнен: 16 коммитов запушены, `main` == `origin/main`.

---

## 5. Правки после аудита (23.07.2026)

Внесены по запросу пользователя. Код этапов B/C по существу не менялся, numeric parity
остаётся действующей.

### 5.1 Тест `StageDMultiFaceInstrumentedTest` → продакшн-координатор
- Переписан: теперь вызывает `OnnxMultiPhotoFaceSwapPipeline.process` целиком (детекция
  один раз, стабильный порядок целей, накопление, per-source кэш эмбеддингов), а не
  реконструирует логику циклом по `OnnxPhotoFaceSwapPipeline`.
- Чтобы сохранить прежние ассерты, координатор теперь отдаёт список per-step
  `pasteRois`. Ассерты не ослаблены: 4 нейролица во фикстуре, T1/T2/T3 изменены тремя
  разными источниками (`pair_01/02/03_source`), **T4 побитово идентичен** (`0`),
  **0 пикселей вне объединения paste ROI**, пересечение T1/T2 содержит второй paste.
- Прогон на `emulator-5554` (API 35 x86_64, CPU): **`OK (1 test)`, `Time: 720.385`**.
  Результат `STAGE_D_MULTI_FACE_RESULT.png` визуально совпадает с прежним (три лица
  заменены, T4 не тронут).

### 5.2 Эмбеддинг не возвращается в ViewModel
- Введён `MultiPhotoFaceSwapResult` (`finalBitmap`, `pasteRois`, три backend, `timings`)
  **без** `sourceEmbedding`; координатор возвращает его. ViewModel и UI читали только
  bitmap/backends/timings — функциональность не изменилась.
- В `finally` дополнительно зануляется финальная копия `sourceEmbedding`
  (`last?.sourceEmbedding?.fill(0f)`) вдобавок к per-source кэшу. Теперь формулировка
  отчёта D «все удерживаемые эмбеддинги зануляются» буквально верна.
- Отчёт D поправлен: §1 и новый §9 «Правки после аудита».

### 5.3 Parity спот-чек одной пары
- Из закреплённого venv (`facefusion-3.7.1-venv`, Python 3.12.10, ORT 1.26.0,
  CPUExecutionProvider) перезапущен `run_facefusion_reference.py` для `pair_01` против
  весов с проверенным SHA-256.
- Сравнение с закоммиченными эталонами `docs/parity/reference/facefusion-3.7.1/pair_01`:
  **все 6 float32-тензоров** (`source_embedding`, `source_embedding_norm`, `raw_output`,
  `raw_mask`, `inswapper_source`, `inswapper_raw_output`) и **все 4 PNG**
  (raw/inswapper/final/box_mask) совпали **побитово (SHA-256)**; landmarks и affine
  matrices совпали до `0.0`. Десктопный эталон воспроизводится детерминированно.

### 5.4 Проверки после правок
| Проверка | Команда | Результат |
| --- | --- | --- |
| Unit tests | `.\gradlew.bat test` | OK — 42 debug + 42 release, 0 failures/errors |
| Lint | `.\gradlew.bat lint` | OK — 0 errors (32 known warnings) |
| Сборка | `.\gradlew.bat assembleDebug assembleDebugAndroidTest` | OK |
| Multi-face instrumentation | `am instrument … StageDMultiFaceInstrumentedTest` | OK — `OK (1 test)`, 720,385 с |
| Parity спот-чек | `run_facefusion_reference.py` pair_01 vs committed | OK — все тензоры/PNG побитово |

- Новый debug-APK после правок: **180 159 399 байт**, SHA-256
  `317EF0E335A53F7D3AD652005565BA81B84FF97CAAE49EC6CA1C01B0BCD2AC9E`. Это **не** хеш из
  отчёта D §2 (`57A3F0F4…`) — тот относился к до-правочной сборке; изменение ожидаемо,
  поскольку менялся продакшн-код.

### 5.5 Разбиение этапа E (ТЗ 2.1)
- `TECHNICAL_SPEC.md` поднят до версии **2.1** (23.07.2026). Этап E §15 разбит на
  **E1 «Качество лица»** (GFPGAN 1.4, parsing/occlusion маска, визуальная сверка) и
  **E2 «Продукт»** (пресеты качества, до/после, прогресс+отмена, MediaStore export,
  cleanup, память).
- В DoD E2 пункты, требующие ARM reference device (12 МП без OOM, 5 циклов памяти,
  thermal), помечены как **отложенные** до появления устройства. Перекрёстные ссылки на
  «этап E» в §7/§8/§11.2 приведены к E1/E2.

Этап E1 **не начат** — жду подтверждения пользователя.
