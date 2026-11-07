# ichkil-go — Arabic Tashkeel for Go

> Vocalize bare Arabic text with a tiny, self-contained ONNX model (≈22 MB, CPU-only,
> no GPU, no tokenizer, no server). One call, fully offline after the first run.

[![CI](https://github.com/Ichkil/ichkil-go/actions/workflows/ci.yml/badge.svg)](https://github.com/Ichkil/ichkil-go/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/Ichkil/ichkil-go.svg)](https://pkg.go.dev/github.com/Ichkil/ichkil-go)
[![Go Version](https://img.shields.io/badge/go-1.21+-blue.svg)](https://go.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face model](https://img.shields.io/badge/%F0%9F%A4%97_model-ichkil%2Fichkil-yellow)](https://huggingface.co/ichkil/ichkil)
[![Try it in your browser](https://img.shields.io/badge/%F0%9F%A4%97_try%20it-ichkil%2Fichkil%2Ddemo-blue)](https://huggingface.co/spaces/ichkil/ichkil-demo)

`ichkil-go` downloads its model from [Hugging Face](https://huggingface.co/ichkil/ichkil)
on first use, verifies its SHA-256, and runs inference with
[ONNX Runtime](https://onnxruntime.ai/). After that it is 100 % offline.

```go
got, err := ichkil.Diacritize(ctx, "محمد قرأ الكتاب")
// got == "مُحَمَّدٌ قَرَأَ الْكِتَابَ"
```

## Install

```bash
go get github.com/Ichkil/ichkil-go
```

Requires Go ≥ 1.21. The only dependency is
[`yalue/onnxruntime_go`](https://github.com/yalue/onnxruntime_go), a thin
cgo wrapper that loads the ONNX Runtime shared library at runtime —
**you need ONNX Runtime ≥ 1.29 installed on the machine** (see below).

### ONNX Runtime on your machine

The Go library loads `onnxruntime` from the system at startup; it never
downloads or bundles the runtime binary.

- **Linux (glibc):** install a package, or download the release build —
  e.g. `libonnxruntime.so.1.x` on `LD_LIBRARY_PATH`, or set
  `ICHKIL_ONNXRUNTIME_LIB=/path/to/libonnxruntime.so` (or
  `libonnxruntime.so.1.x`) to point at an exact file.
- **macOS:** `libonnxruntime.dylib` — same override variable works.
- **Windows:** `onnxruntime.dll` on `PATH` — or
  `ICHKIL_ONNXRUNTIME_LIB=C:\path\to\onnxruntime.dll`.
- **Easiest everywhere:** `pip install onnxruntime` and point the env var at it:

  ```bash
  ICHKIL_ONNXRUNTIME_LIB=$(python -c "import onnxruntime; print(onnxruntime.get_library_path())")
  ```

The library resolves in this order: `$ICHKIL_ONNXRUNTIME_LIB` (explicit path),
then the standard system lookup for the platform's ONNX Runtime library.

## Quickstart

```go
package main

import (
    "context"
    "log"

    "github.com/Ichkil/ichkil-go"
)

func main() {
    ctx := context.Background()

    // one-liner (lazy model download on first call, then cached)
    got, err := ichkil.Diacritize(ctx, "العلم نور والجهل ظلمة")
    if err != nil {
        log.Fatal(err)
    }
    log.Println(got) // الْعِلْمُ نُورٌ وَالْجَهْلُ ظُلْمَةٌ

    // explicit control
    d, err := ichkil.New(ctx) // default repository: ichkil/ichkil@main
    if err != nil {
        log.Fatal(err)
    }
    defer d.Close()

    text, _ := d.Diacritize("محمد قرأ الكتاب")
    labels, _ := d.PredictLabels("كتاب") // one label id per character, 0..15

    // fully offline: point at local files
    d2, err := ichkil.NewFromFiles("./model.onnx")
    _ = d2
}
```

A `Diacritizer` is safe for concurrent use; `Close` releases the ONNX Runtime
session. Inference is synchronous CPU work — the `context.Context` is used for
model resolution (network) only.

### Command line

```bash
go run github.com/Ichkil/ichkil-go/cmd/ichkil "محمد قرأ الكتاب"
# مُحَمَّدٌ قَرَأَ الْكِتَابَ

go run github.com/Ichkil/ichkil-go/cmd/ichkil --json "العلم نور"
# {"input":"العلم نور","output":"الْعِلْمُ نُورٌ"}

cat text.txt | ichkil                    # read from stdin
ichkil --model ./model.onnx "محمد"      # offline mode
```

## API

| Item | Description |
| --- | --- |
| `ichkil.Diacritize(ctx, text) (string, error)` | Vocalize text (module-level convenience, lazy default model). |
| `ichkil.PredictLabels(ctx, text) ([]int, error)` | Label id (0–15) per character. |
| `ichkil.New(ctx, opts...) (*Diacritizer, error)` | Resolve the model from Hugging Face and open an ONNX session. |
| `ichkil.NewFromFiles(modelPath, opts...) (*Diacritizer, error)` | Build from a local `model.onnx` (+ `config.json` beside it). |
| `(*Diacritizer).Diacritize(text)` / `.PredictLabels(text)` | Instance methods (same behavior). |
| Options | `WithRepoID`, `WithRevision`, `WithCacheDir`, `WithToken`, `WithNumThreads`, `WithoutChecksumVerification`, `WithBaseURL`, `WithConfigPath`. |
| `ichkil.SetDefault(d)` / `ResetDefault()` | Manage the shared default instance used by the module-level API. |
| `ichkil.RuntimeVersion()` | Version of the loaded ONNX Runtime library. |
| Pipeline | `NormalizeText`, `StripDiacritics`, `Encode`, `Decode`, `Attach`, `Argmax`, `IsArabicLetter`, label tables. |
| Errors | Sentinels: `ErrChecksumMismatch`, `ErrModelLoad`, `ErrInference` (match with `errors.Is`). |
| Env | `HF_TOKEN`, `HF_ENDPOINT`, `ICHKIL_CACHE_DIR`, `ICHKIL_ONNXRUNTIME_LIB`. |

### Label contract

The model predicts exactly one label per input character:

| id | meaning | id | meaning |
| --- | --- | --- | --- |
| 0 | non-letter token / padding | 7 | shadda + fatha (ّـَ) |
| 1 | letter, no diacritic | 8 | shadda + damma (ّـُ) |
| 2 | fatha (ـَ) | 9 | shadda + kasra (ّـِ) |
| 3 | damma (ـُ) | 10 | tanwin fatha (ـً) |
| 4 | kasra (ـِ) | 11 | tanwin damma (ـٌ) |
| 5 | sukun (ـْ) | 12 | tanwin kasra (ـٍ) |
| 6 | shadda (ـّ) | 13 | shadda + tanwin fatha (ـًّ) |
| | | 14 | shadda + tanwin damma (ـٌّ) |
| | | 15 | shadda + tanwin kasra (ـٍّ) |

Input is normalized first (Unicode NFC, ٱ→ا, ی→ي, ک→ك, zero-width characters
removed, whitespace collapsed) and existing marks are stripped, so vocalized and
bare input give the same result. Labels predicted on non-letters are
unspecified and never attached.
and re-applying it changes nothing. Non-Arabic characters (Latin letters,
digits, punctuation, spaces) pass through unchanged.

## Model provenance

- Repository: [`ichkil/ichkil`](https://huggingface.co/ichkil/ichkil), pinned to the
  **`v0.3.0`** tag by default (revision `v0.1.0` loads the previous 13-label model).
- SHA-256 of `model.onnx` (v0.3.0): `9779e1aca89cebc77a9b92037b7d700c422cdf6c3774d4883f4652f641f3f705`
- Contract: `input_ids` int64 `[batch, seq]` → `logits` float32 `[batch, seq, 16]`;
  dynamic length, tested up to 1024 characters per call.
- Quality (v0.3.0): DER 2.32 % on Fadel et al. 2019 (classical), 2.14 % on WikiNews
  (modern Arabic); the ONNX artifact scores identically to the PyTorch reference.