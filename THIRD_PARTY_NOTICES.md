# Third-Party Notices

Дата проверки: 19.07.2026.

FaceSwapLocal не содержит весов нейросетевых моделей ни в Git, ни в APK. Перечисленные
ниже модели используются только после явного локального импорта пользователем через
системный picker и проверки SHA-256. Эти notices документируют происхождение и не
предоставляют дополнительных прав на веса.

## ONNX Runtime Android 1.26.0

Источник: <https://github.com/microsoft/onnxruntime/tree/v1.26.0>
Полный файл лицензии: <https://github.com/microsoft/onnxruntime/blob/v1.26.0/LICENSE>

ONNX Runtime распространяется по MIT License. Дословный текст лицензии:

```text
MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## FaceFusion 3.7.1 — эталон препроцессинга

Источник: <https://github.com/facefusion/facefusion/tree/3.7.1>
Файл лицензии: <https://github.com/facefusion/facefusion/blob/3.7.1/LICENSE.md>

Файл `LICENSE.md` tag 3.7.1 дословно содержит только:

```text
OpenRAIL-AS license

Copyright (c) 2026 Henry Ruhs
```

Полного текста OpenRAIL-AS в этом файле нет. Поэтому FaceSwapLocal не делает вывода о
дополнительных правах из одного названия лицензии. Python-код FaceFusion в приложение
не включён; собственная Kotlin-реализация сверена с порядком операций, tensor
contracts, числовыми шаблонами и результатами FaceFusion 3.7.1.

## HyperSwap 1a 256

Файл модели:
<https://huggingface.co/facefusion/models-3.3.0/blob/6271ab1c4619ceb2e534d65e0ec973f22a9aec38/hyperswap_1a_256.onnx>
Каталог FaceFusion:
<https://github.com/facefusion/facefusion/blob/3.7.1/facefusion/processors/modules/face_swapper/core.py>

Дословные metadata каталога FaceFusion 3.7.1:

```text
vendor: FaceFusion
license: ResearchRAIL
year: 2025
```

В `facefusion/models-3.3.0` на зафиксированной ревизии нет полного текста, версии или
URL конкретной ResearchRAIL-лицензии. Поэтому право личного использования за пределами
исследовательской оценки и право распространения не считаются подтверждёнными. Вес не
включается в репозиторий или APK.

## InsightFace: ArcFace W600K R50 и InSwapper 128 fp16

Файлы reference-набора:

- <https://huggingface.co/facefusion/models-3.0.0/blob/728b9659bd9691bf32cbf7f61af478d94b7ba81e/arcface_w600k_r50.onnx>;
- <https://huggingface.co/facefusion/models-3.0.0/blob/728b9659bd9691bf32cbf7f61af478d94b7ba81e/inswapper_128_fp16.onnx>.

Официальная политика: <https://github.com/deepinsight/insightface#license>.

Из официального README InsightFace дословно:

```text
The code of InsightFace is released under the MIT License.
```

Для pretrained weights официальный README отдельно использует формулировку
`available for non-commercial research purposes only`; MIT-лицензия к весам не
применяется. В обновлении от 24.11.2025 InsightFace просит обращаться по адресу
`contact@insightface.ai` для лицензирования серии InSwapper и по адресу
`recognition-oss-pack@insightface.ai` для открытых recognition-моделей.

FaceFusion 3.7.1 дословно маркирует обе используемые модели как:

```text
license: Non-Commercial
```

Веса не включаются в репозиторий или APK. Локальный импорт не отменяет ограничение на
некоммерческое исследовательское использование и не заменяет отдельную лицензию
InsightFace.

## YOLOFace 8n

Файл модели:
<https://huggingface.co/facefusion/models-3.0.0/blob/728b9659bd9691bf32cbf7f61af478d94b7ba81e/yoloface_8n.onnx>
Каталог FaceFusion:
<https://github.com/facefusion/facefusion/blob/3.7.1/facefusion/face_detector.py>

FaceFusion 3.7.1 дословно указывает:

```text
vendor: derronqi
license: GPL-3.0
year: 2022
```

Встроенные ONNX metadata того же файла дословно содержат:

```text
author: Ultralytics
license: AGPL-3.0 https://ultralytics.com/license
version: 8.1.0
```

Из-за расхождения `GPL-3.0` и `AGPL-3.0`, а также отсутствия отдельного полного
лицензионного текста на странице файла проект применяет более консервативное правило:
вес не распространяется и устанавливается только локальным импортом после проверки
SHA-256.

## Операционное решение проекта

- В репозитории действует исключение `*.onnx`.
- APK содержит ONNX Runtime, но не содержит `yoloface_8n`,
  `arcface_w600k_r50`, `hyperswap_1a_256` или `inswapper_128_fp16`.
- Приложение не имеет `INTERNET` и не скачивает веса.
- Для каждого веса разрешён только явный picker-import; приложение принимает лишь
  размер и SHA-256 из `docs/MODEL_CARD.md`.
- До получения однозначных условий ни один вес нельзя публиковать вместе с исходниками,
  APK или иным дистрибутивом FaceSwapLocal.

Эта сводка фиксирует проверенные notices и техническое решение проекта, но не является
юридической консультацией.
