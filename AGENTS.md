# FaceSwapLocal instructions

## Product goal

Build a personal Android app that replaces one or multiple faces in photos and,
after the photo MVP is reliable, in videos. All inference and media processing
must happen on the device.

## Non-negotiable constraints

- Never add an `INTERNET` permission, analytics, telemetry, accounts, or uploads.
- Use Kotlin, Jetpack Compose, coroutines, and native Android APIs.
- Keep `minSdk` at 28 unless a documented model/runtime requirement changes it.
- Treat every imported image or video as private user data.
- Delete temporary frames and intermediate files after success, cancellation, or failure.
- Do not commit neural model weights until their source and license are documented.
- Do not claim the neural swap is implemented while the engine is still a placeholder.

## Delivery order

1. Photo selection and local multi-face detection.
2. Explicit target-face to source-face mapping.
3. Licensed on-device identity encoder and swap model.
4. Masking, color matching, restoration, and photo export.
5. Video decoding, face tracking, temporal stabilization, and encoding.

## Architecture

- `domain`: immutable models, assignment rules, and processing contracts.
- `data`: Android bitmap decoding and bundled ML Kit face detection.
- `ui`: Compose screen and state holder.
- Future model/runtime code should live under `inference` and implement
  `FaceSwapEngine` without leaking runtime-specific types into the UI.

## Useful commands

```bash
./gradlew test
./gradlew assembleDebug
```

Before finishing a change, run both commands when an Android SDK is available.

