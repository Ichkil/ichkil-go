package ichkil

import "errors"

// Sentinel errors returned by the ichkil package. Use errors.Is to test for
// them, e.g. errors.Is(err, ichkil.ErrChecksumMismatch).
var (
	// ErrChecksumMismatch is returned when the SHA-256 of model.onnx does not
	// match the expected digest (from SHA256SUMS or config.sha256).
	ErrChecksumMismatch = errors.New("ichkil: model checksum mismatch")

	// ErrModelLoad is returned when the model artifact or its config is
	// missing, invalid or unreadable (including ONNX Runtime environment
	// initialization failures).
	ErrModelLoad = errors.New("ichkil: model load error")

	// ErrInference is returned when an ONNX Runtime inference fails.
	ErrInference = errors.New("ichkil: inference error")
)
