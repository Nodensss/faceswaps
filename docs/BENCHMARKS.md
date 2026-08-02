# Benchmarks FaceSwapLocal

## E1, контрольная точка 1: two-pass coordinator, API 35 x86_64

Дата: 2026-08-02. Вход `stage_d_group_target.png` 1600×1100, три назначения
InSwapper (T1/T2/T3), T4 — `Не менять`. ONNX Runtime Android 1.26.0, CPU,
`airplane_mode_on=1`.

| Прогон | Состав | Время coordinator | Пространственные инварианты |
| --- | --- | ---: | --- |
| Сила `0` | 3 swap, GFPGAN/BiSeNet не открывались | 170 439 ms | вне union ROI: 0; T4: 0 изменений |
| Сила `0.8` | 3 swap, затем 3× GFPGAN + 3× BiSeNet | 322 886 ms | вне union(swap, enhance) ROI: 0; T4: 0 изменений |
| Сила `1.0` | 3 swap, затем 3× GFPGAN + 3× BiSeNet | 328 106 ms | вне union(swap, enhance) ROI: 0; T4: 0 изменений |

Полный instrumentation runner с подготовкой и девятью финальными ArcFace embeddings:
887,171 с, `OK (1 test)`. Журнал реальных open/close событий подтвердил, что последняя
InSwapper session закрылась до первой GFPGAN session; максимум одновременно открытых
среди InSwapper/GFPGAN/BiSeNet — `1`. Ожидаемый source остался ближайшим для T1/T2/T3
при всех трёх силах. Peak heap и thermal не измерялись; AVD не является reference device.
Потенциальный FFHQ ROI неназначенного T4 также передан compositor как глобальная
no-write область. Полные числа:
`docs/parity/android/api35-x86_64/checkpoint_1/checkpoint_1_results.json`.

| Baseline ArcFace margin | Сила `0` | Сила `0.8` | Сила `1.0` |
| --- | ---: | ---: | ---: |
| T1→S1 | 0,520990 | 0,485556 | 0,465453 |
| T2→S2 | 0,723044 | 0,665249 | 0,645125 |
| T3→S3 | 0,600909 | 0,526425 | 0,502646 |

Это неизменяемая AVD-база. На первом reference device таблица повторяется на разрешённом
реальном фото отдельным набором результатов; подменять ею эти значения нельзя.

## Этап D: multi-face fixture, API 35 x86_64

Дата: 2026-07-23. Вход `stage_d_group_target.png`: 1600×1100, четыре
детектированных лица, три последовательных назначения InSwapper, T4 без изменения.
Backend: ONNX Runtime Android 1.26.0 CPU; авиарежим включён.

| Проверка | Результат |
| --- | --- |
| Полный instrumentation test | 316,565 с (`OK (1 test)`) |
| Порядок | одна target-детекция по оригиналу, затем T1 → T2 → T3 поверх накопленного результата |
| Identity embedding | один ArcFace embedding на уникальный source в рамках задачи; cache очищен и массивы обнулены в `finally` |
| Геометрия | paste ROI T1/T2 пересекаются; в пересечении остаётся второй paste |
| Неизменяемые области | T4 побитово идентичен; вне объединения трёх paste ROI изменено 0 пикселей |
| Артефакт | `docs/reports/img/STAGE_D_MULTI_FACE_RESULT.png` |

Peak Java/native heap и thermal status в этом прогоне не снимались; AVD не является
reference device и результат времени нельзя переносить на ARM-телефон.

Дата измерений этапа B: 2026-07-19. Дата измерений этапа C: 2026-07-21.

## Статус reference device

Реальный Android-телефон пользователя в этапе B не был подключён. Поэтому AVD ниже
не объявляется reference device, а его результаты нельзя использовать как прогноз
времени или расхода памяти на ARM-телефоне. После первого подключения телефона сюда
нужно записать модель, Android, SoC, RAM и backend без серийного номера, IMEI и иных
идентификаторов и повторить сценарии этого документа.

## Конфигурации

### Android, предварительная конфигурация

| Параметр | Значение |
| --- | --- |
| Тип | Android Emulator / AVD `QuizMaster_API35` |
| Модель | `Android SDK built for x86_64` |
| Android | 15 / API 35 |
| ABI / hardware | x86_64 / `ranchu` |
| CPU | 2 virtual cores; физический SoC отсутствует |
| RAM | `MemTotal: 2019876 kB` |
| Runtime | ONNX Runtime Android 1.26.0 |
| Backend parity | CPU, 1 intra-op thread на этом AVD |
| Экран AVD | 1080×1920 |

Команды фиксации конфигурации:

```powershell
adb shell getprop ro.build.version.release
adb shell getprop ro.build.version.sdk
adb shell getprop ro.product.model
adb shell getprop ro.product.cpu.abi
adb shell getprop ro.hardware
adb shell getprop ro.boot.qemu.avd_name
adb shell cat /proc/meminfo
adb shell nproc
```

### Десктопный эталон

| Параметр | Значение |
| --- | --- |
| ОС | Windows 10.0.19045 |
| CPU | Intel Core i5-4200M @ 2,50 GHz, 2 cores / 4 logical processors |
| RAM | 17 123 532 800 bytes |
| FaceFusion | 3.7.1, commit `3f81a8a78454089d720b8f318a12ae1702c4633b` |
| Runtime | Python 3.12.10, ONNX Runtime 1.26.0 |
| Backend | `CPUExecutionProvider`, до 4 intra-op threads |

## Контрольный сценарий этапа B

- три синтетические пары из `docs/parity/inputs`;
- source и target каждого прогона: 1254×1254, по одному найденному лицу;
- режим: сырой кроп этапа B, HyperSwap 256×256 либо InSwapper 128×128;
- Android-последовательность: два запуска detector в одной session, затем recognizer,
  затем один swapper; тяжёлые Android sessions не открыты одновременно;
- blending, inverse transform и restoration не выполняются, поэтому время blending
  в таблице отмечено `n/a`;
- identity embedding InSwapper: `embedding @ emap / l2_norm(embedding)`;
- все Android-измерения получены внутри instrumentation через
  `SystemClock.elapsedRealtimeNanos()`.

Команда Android-теста приведена в `docs/parity/README.md`. Полные машинно-читаемые
результаты: `docs/parity/android/api35-x86_64/android_results.json`.

## Android CPU: результаты по каждому прогону

`Detector` включает source и target. `Total` также включает alignment, tensor
подготовку, проверку файла модели и преобразование сырого выхода в Bitmap.

| Пара | Swapper | Detector, ms | Recognizer, ms | Swapper, ms | Blending | Total, ms | Raw SSIM | Визуальное наблюдение |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `pair_01` | HyperSwap 1a 256 | 2 689 | 3 411 | 18 138 | n/a | 24 482 | 0,998060 | перенос identity слабый, совпадает с desktop |
| `pair_01` | InSwapper 128 fp16 | 2 550 | 3 706 | 33 474 | n/a | 39 766 | 0,998275 | identity source узнаваема |
| `pair_02` | HyperSwap 1a 256 | 2 506 | 3 655 | 16 517 | n/a | 22 777 | 0,997760 | перенос identity слабый, совпадает с desktop |
| `pair_02` | InSwapper 128 fp16 | 2 813 | 3 974 | 33 369 | n/a | 40 212 | 0,997551 | identity source узнаваема |
| `pair_03` | HyperSwap 1a 256 | 2 420 | 3 758 | 16 238 | n/a | 22 514 | 0,997513 | перенос identity слабый, совпадает с desktop |
| `pair_03` | InSwapper 128 fp16 | 2 201 | 4 270 | 32 799 | n/a | 39 310 | 0,997645 | identity source узнаваема |

Медианы на трёх парах:

| Swapper | Detector, ms | Recognizer, ms | Swapper, ms | Total, ms |
| --- | ---: | ---: | ---: | ---: |
| HyperSwap 1a 256 | 2 506 | 3 655 | 16 517 | 22 777 |
| InSwapper 128 fp16 | 2 550 | 3 974 | 33 369 | 39 766 |

Эти медианы являются только первой регрессионной точкой для данного AVD. Правило
§12 о допустимом ухудшении не более 20% следует применять к повторному benchmark на
той же конфигурации, а не сравнивать AVD с будущим ARM reference device.

### Ручной UI smoke test

После принудительного выбора `CPU_ONLY` полный сценарий через Compose UI и системный
Photo Picker успешно завершился в авиарежиме на `pair_01`: ML Kit нашёл по одному
лицу, назначение `Целевое лицо 1 → Источник 1` сохранилось, а InSwapper показал
узнаваемый сырой source identity. Первый холодный UI-прогон дал detector 6 160 ms,
ArcFace 9 609 ms, swapper 92 114 ms, всего 108 233 ms. Повторный прогон 21.07.2026
после переустановки финального APK дал соответственно 11 326, 18 611, 154 649 и
185 719 ms. Оба значения включают обязательное повторное хеширование файлов перед
session; разброс отражает нестабильную нагрузку x86_64 AVD и не заменяет контролируемые
медианы instrumentation выше.
Скриншот результата: `docs/reports/img/STAGE_B_RAW_CROP_API35.png`.

## Desktop CPU: reference timings

Значения ниже записаны самим `run_facefusion_reference.py`; детекция source и target
сохранена раздельно. Они нужны для воспроизводимости parity, а не для сравнения
производительности с Android Emulator.

| Пара | Source detect, ms | Target detect, ms | Recognizer, ms | HyperSwap, ms | InSwapper, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pair_01` | 458,576 | 296,891 | 468,096 | 5 937,275 | 8 711,342 |
| `pair_02` | 306,319 | 333,815 | 784,312 | 2 216,512 | 8 689,646 |
| `pair_03` | 377,861 | 497,419 | 820,568 | 2 170,693 | 7 185,345 |

Эти значения получены усиленным reference runner, который вызывает production-
функции FaceFusion с инструментированными локальными ORT-сессиями. Первый HyperSwap-
прогон включает холодный старт CPU graph; target embedding рассчитывается FaceFusion,
но столбец `Recognizer` сохраняет только время source identity embedding.

## Память и thermal

Пиковые Java heap и native heap во время inference не измерялись Android Studio
Profiler, поэтому соответствие ориентиру ~1,5 GB на reference device на этапе B не
заявляется. После instrumentation Android сохранил точечное значение `rss=676MB` при
штатной остановке процесса runner (`reason=USER REQUESTED`, `finished inst`); это не
peak heap и не заменяет профилирование. Команда диагностики:

```powershell
adb shell dumpsys activity exit-info com.faceswaplocal.app
```

Thermal status во время прогонов не семплировался. После теста эмулятор возвращал
симулированные `Thermal Status: 0` и 30,8 °C; эти значения не описывают нагрев
реального устройства и в benchmark не засчитываются.

## XNNPACK на API 35 x86_64

Отдельный ручной UI-прогон InSwapper с запрошенным XNNPACK завершился native crash
19.07.2026 в 06:37:23 UTC. Crash buffer фиксирует `SIGABRT` в
`libonnxruntime.so` во время `OrtEnvironment.createSession`; ActivityManager фиксирует
`reason=APP CRASH(NATIVE)`, `status=6`, `rss=186MB`. Это не диагностировано как OOM:
доказательств OOM в логе нет.

```text
F/libc: Fatal signal 6 (SIGABRT) ... in tid DefaultDispatch
F/DEBUG: Cmdline: com.faceswaplocal.app
F/DEBUG: #01 ... libonnxruntime.so
F/DEBUG: #17 ... OrtSession_createSession
F/DEBUG: #42 ... OnnxRawFaceSwapPipeline.runWithSession
```

Команды, которыми подтверждён инцидент:

```powershell
adb logcat -d -b crash -v time -t 120
adb shell dumpsys activity exit-info com.faceswaplocal.app
```

Native abort завершает весь процесс и не может быть перехвачен Kotlin-кодом как
`OrtException`, поэтому автоматический exception fallback не успевает выполниться.
Для пользовательского пути этапа B выбран гарантированный `CPU_ONLY`. Возможность
создать XNNPACK session сохранена для отдельной проверки на ARM reference device, но
`x86`/`x86_64` теперь превентивно маршрутизируются в `CPU_FALLBACK` до нативного
`createSession`; до ARM-проверки XNNPACK не считается поддержанным backend. Этот вывод
относится только к ONNX Runtime 1.26.0 на данном x86_64 AVD и не переносится на
реальные устройства.

## Что измерить на первом reference device

1. Модель телефона, Android, SoC, RAM и runtime/backend без уникальных идентификаторов.
2. Те же шесть сырых прогонов CPU, затем изолированную проверку XNNPACK.
3. Peak Java/native heap через Profiler или серию `dumpsys meminfo` во время каждого
   шага, а не только после него.
4. Thermal status до, во время и после inference.
5. Повторный цикл не менее пяти раз для проверки устойчивого роста heap.
6. После этапа C — отдельные времена inverse transform и blending; после этапа E —
   restoration и полное время 12 MP сценария.

## Контрольный сценарий этапа C

Три синтетические пары 1254×1254 обработаны последовательно через InSwapper 128 fp16
на CPU. Android total включает detector, recognizer, swapper, inverse transform,
masked color matching и compositing. `Compositing` включает постобработку после
swapper. AVD не является reference device, поэтому эти времена нельзя переносить на
ARM-телефон.

| Пара | Detector, ms | Recognizer, ms | Swapper, ms | Compositing, ms | Total, ms | Full SSIM | ROI SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pair_01` | 2 918 | 3 969 | 38 752 | 1 907 | 47 760 | 0,999696 | 0,999606 |
| `pair_02` | 2 391 | 6 298 | 39 847 | 942 | 49 542 | 0,998797 | 0,997917 |
| `pair_03` | 2 338 | 4 178 | 39 493 | 923 | 46 978 | 0,997602 | 0,997305 |
| **Медиана** | **2 391** | **4 178** | **39 493** | **942** | **47 760** | — | — |

Desktop FaceFusion production `swap_face` CPU timing: `4 780,079`, `5 689,506` и
`5 645,149` ms; медиана — `5 645,149` ms. Desktop и AVD timings не сравниваются как
производительность: различаются архитектура, ORT build и окружающий pipeline.

Ручной холодный UI-прогон `pair_01` в авиарежиме дал detector `16 406` ms, ArcFace
`23 593` ms, swapper `211 152` ms, compositing `9 229` ms, total `262 232` ms.
Все три backend в UI подтвердили `CPU_FALLBACK`. Скриншоты:
`docs/reports/img/STAGE_C_FINAL_API35.png` и
`docs/reports/img/STAGE_C_FINAL_DETAILS_API35.png`.

Документированное отличие: canonical FaceFusion frame не содержит отдельного color
matching, тогда как Android применяет masked RGB mean/std с strength `0.65` по
FR-PHOTO-06. Измеренный эффект включён в SSIM/MAE; вне inverse paste ROI изменено
0 пикселей во всех трёх парах.

## Изолированное parity-ядро E1

Дата: 01.08.2026. AVD API 35 x86_64, ONNX Runtime Android 1.26.0,
`CPUExecutionProvider`, один intra-op поток, `airplane_mode_on=1`. Вход — один и тот
же закоммиченный desktop FaceFusion crop 512×512; detector, swapper, coordinator,
compositor и UI не запускались.

| Операция | Время Android, ms | Desktop CPU, ms | Численная проверка |
| --- | ---: | ---: | --- |
| GFPGAN 1.4 raw inference | 47 288 | 12 828 | SSIM `0.9999999999997585`, MAE `1.0948e-7` |
| BiSeNet ResNet-34 main output + argmax | 6 838 | 1 417 | class agreement `1.0`, protected-region IoU `1.0` |

Android-время включает только вызов соответствующего метода core, в том числе полную
SHA-256-проверку приватного model file и создание/закрытие CPU session; desktop-время
в metadata отражает только inference. Поэтому значения не являются прямым сравнением
скорости. Peak heap и thermal не измерялись; AVD не считается reference device.
