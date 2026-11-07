# Changelog

All notable changes to the `ichkil-go` module are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-11-07

### Changed

- Targets model **v0.3.0**: `DefaultRevision` is now pinned to the `v0.3.0` tag of
  `ichkil/ichkil` (was `main`), so a future model release can never change the
  behavior of an installed version. `WithRevision("v0.1.0")` still loads the old model.
- **16 labels**: adds 13/14/15 = shadda + tanwin fatha / damma / kasra
  (e.g. عَدُوٌّ, قَوِيًّا). Ids 0–12 are unchanged.
- `StripDiacritics` now normalizes first (new `NormalizeText`: NFC, ٱ→ا, ی→ي, ک→ك,
  ہ→ه, zero-width characters removed, whitespace collapsed), matching the model's
  training data.
- Arabic letter range is now U+0621–U+063A, U+0641–U+064A (tatweel is not a letter).
- Model quality: DER 2.32 % on Fadel (was 2.80 %), 2.14 % on modern news (was ~11 %);
  much better phrase endings, proper nouns and loanwords. Model size ≈22 MB.
- Golden vectors updated to the v0.3.0 contract (37 vectors).
- `PredictLabels` returns one label per character of the normalized, stripped
  text (whitespace-only input now yields no labels).
- New dependency: `golang.org/x/text` (Unicode NFC normalization).

## [1.0.0] - 2026-01-01

### Added

- Initial public release.
- `Diacritizer` — one-call Arabic Tashkeel backed by a ≈5 MB ONNX model
  (`input_ids` int64 `[batch, seq]` → `logits` float32 `[batch, seq, 13]`),
  downloaded from Hugging Face (`ichkil/ichkil`), SHA-256-verified, cached
  locally (atomic `.part` → rename); `NewFromFiles()` for fully offline use.
- Functional-options API: `WithRepoID`, `WithRevision`, `WithCacheDir`,
  `WithToken`, `WithNumThreads`, `WithoutChecksumVerification`, `WithBaseURL`,
  `WithConfigPath`.
- Module-level convenience API: `ichkil.Diacritize(ctx, text)`,
  `ichkil.PredictLabels(ctx, text)`, `ichkil.SetDefault(d)`,
  `ichkil.ResetDefault()`.
- Pure text pipeline: `StripDiacritics`, `Encode`, `Decode`, `Attach`, `Argmax`,
  `IsArabicLetter`, label tables (13 labels), tatweel normalization — shared with
  and kept in sync with the Python and JS runtimes.
- Sentinel errors `ErrChecksumMismatch`, `ErrModelLoad`, `ErrInference`
  (matchable with `errors.Is`).
- CLI `cmd/ichkil` (positional text or stdin, `--model` offline mode, `--json`,
  `--repo`, `--revision`, `--version`; exit codes 0/1/2).
- ONNX Runtime shared library resolved at runtime from the system — override
  with `ICHKIL_ONNXRUNTIME_LIB`; no bundled binaries in the module.
- Cross-runtime golden vector suite (`testdata/golden.json`, copied from the
  core repository) plus pure-pipeline, download-verification, and integration
  tests (integration tests skip automatically when ONNX Runtime is absent).
- CI: gofmt + `go vet` + `go build`, test matrix across Linux/macOS/Windows,
  minimum Go version check.
- Publishing: `v*` tags (semver) — the module is consumed via
  proxy.golang.org / `go get github.com/Ichkil/ichkil-go@vX.Y.Z`; the Publish
  workflow verifies the tag, tests the tagged commit, and creates a GitHub
  Release.
