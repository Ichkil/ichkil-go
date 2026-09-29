// Package ichkil provides Arabic Tashkeel (diacritization): it assigns
// harakat, shadda and tanwin to bare Arabic text.
//
// The engine is a tiny, self-contained ONNX model (≈5 MB, CPU-only) that is
// downloaded from Hugging Face on first use and cached locally; afterwards
// everything works offline.
//
// Quick start:
//
//	// One-liner — downloads + caches the model on first use.
//	got, err := ichkil.Diacritize(ctx, "محمد قرأ الكتاب")
//	// got == "مُحَمَّدٌ قَرَأَ الْكِتَابِ"
//
// Or, for explicit control:
//
//	d, err := ichkil.New(ctx)                    // default repository
//	defer d.Close()
//	got, err := d.Diacritize("الكتاب على الطاولة")
//
//	d, err := ichkil.NewFromFiles("model.onnx")  // fully offline
//
// A Diacritizer is safe for concurrent use. Close releases the underlying
// ONNX Runtime session; it must not be called concurrently with other
// methods.
package ichkil

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// Model input / output names (fixed by the exported ONNX graph).
const (
	inputName  = "input_ids"
	outputName = "logits"
)

// config is the normalized option set shared by New, NewFromFiles and
// ResolveModel.
type config struct {
	repoID         string
	revision       string
	cacheDir       string
	token          string
	verifyChecksum bool
	numThreads     int
	hfBaseURL      string
	configPath     string
}

func defaultConfig() config {
	return config{
		repoID:         DefaultRepoID,
		revision:       DefaultRevision,
		cacheDir:       defaultCacheDir(),
		token:          os.Getenv("HF_TOKEN"),
		verifyChecksum: true,
		hfBaseURL:      hfBaseURL(),
	}
}

// Option customizes model resolution and the ONNX Runtime session.
type Option func(*config)

// WithRepoID sets the Hugging Face model repository (default "ichkil/ichkil").
func WithRepoID(repoID string) Option {
	return func(c *config) { c.repoID = repoID }
}

// WithRevision sets the branch, tag or commit to fetch (default "main").
func WithRevision(revision string) Option {
	return func(c *config) { c.revision = revision }
}

// WithCacheDir sets the model cache directory (default $ICHKIL_CACHE_DIR or
// ~/.cache/ichkil).
func WithCacheDir(dir string) Option {
	return func(c *config) { c.cacheDir = dir }
}

// WithToken sets the Hugging Face token used for gated / private repositories
// (default: $HF_TOKEN).
func WithToken(token string) Option {
	return func(c *config) { c.token = token }
}

// WithNumThreads sets the ONNX Runtime intra-op thread count (0 = ORT
// heuristic, the default).
func WithNumThreads(n int) Option {
	return func(c *config) { c.numThreads = n }
}

// WithoutChecksumVerification disables the SHA-256 verification of
// model.onnx against SHA256SUMS (enabled by default).
func WithoutChecksumVerification() Option {
	return func(c *config) { c.verifyChecksum = false }
}

// WithBaseURL sets the Hugging Face base URL (default: $HF_ENDPOINT or
// https://huggingface.co).
func WithBaseURL(baseURL string) Option {
	return func(c *config) { c.hfBaseURL = baseURL }
}

// WithConfigPath sets the config.json used by NewFromFiles (default:
// config.json next to the model).
func WithConfigPath(path string) Option {
	return func(c *config) { c.configPath = path }
}

// Diacritizer is a ready-to-use Arabic diacritizer backed by ONNX Runtime.
type Diacritizer struct {
	repoID    string
	revision  string
	sym2ID    map[string]int
	unkID     int
	numLabels int
	maxSeqLen int

	session  *ort.DynamicAdvancedSession
	sessOpts *ort.SessionOptions
	mu       sync.Mutex // guards Close only
}

// New creates a Diacritizer by resolving the model from Hugging Face (cached
// after the first download) and opening an ONNX Runtime session.
func New(ctx context.Context, opts ...Option) (*Diacritizer, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	bundle, err := ResolveModel(ctx, &cfg)
	if err != nil {
		return nil, err
	}
	return newFromPaths(bundle.OnnxPath, bundle.Config, cfg.numThreads, bundle.RepoID, bundle.Revision)
}

// NewFromFiles creates a Diacritizer from local model.onnx + config.json
// files (fully offline). The config defaults to config.json next to the
// model; use WithConfigPath to point elsewhere.
func NewFromFiles(modelPath string, opts ...Option) (*Diacritizer, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	if fi, err := os.Stat(modelPath); err != nil || !fi.Mode().IsRegular() {
		return nil, fmt.Errorf("%w: model file not found: %s", ErrModelLoad, modelPath)
	}
	configPath := cfg.configPath
	if configPath == "" {
		configPath = filepath.Join(filepath.Dir(modelPath), "config.json")
	}
	config, err := LoadConfig(configPath)
	if err != nil {
		return nil, err
	}
	return newFromPaths(modelPath, config, cfg.numThreads, "<local>", filepath.Base(modelPath))
}

// newFromPaths opens an ONNX Runtime session on the given model file.
func newFromPaths(onnxPath string, config Config, numThreads int, repoID, revision string) (*Diacritizer, error) {
	if err := ensureEnvironment(); err != nil {
		return nil, err
	}
	sessOpts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrModelLoad, err)
	}
	sessOpts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll)
	if numThreads > 0 {
		if err := sessOpts.SetIntraOpNumThreads(numThreads); err != nil {
			sessOpts.Destroy()
			return nil, fmt.Errorf("%w: %v", ErrModelLoad, err)
		}
	}
	session, err := ort.NewDynamicAdvancedSession(onnxPath, []string{inputName}, []string{outputName}, sessOpts)
	if err != nil {
		sessOpts.Destroy()
		return nil, fmt.Errorf("%w: failed to load ONNX model %s: %v", ErrModelLoad, onnxPath, err)
	}
	return &Diacritizer{
		repoID:    repoID,
		revision:  revision,
		sym2ID:    config.Sym2ID,
		unkID:     int(sym2IDDefault(config.Sym2ID, "<unk>", 1)),
		numLabels: config.NumTags,
		maxSeqLen: config.MaxSeqLen,
		session:   session,
		sessOpts:  sessOpts,
	}, nil
}

func sym2IDDefault(sym2ID map[string]int, key string, fallback int) int {
	if v, ok := sym2ID[key]; ok {
		return v
	}
	return fallback
}

// Close releases the ONNX Runtime session. The diacritizer must not be used
// after Close; Close is idempotent. The shared ONNX Runtime environment
// (process-wide) is left intact.
func (d *Diacritizer) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.session == nil {
		return nil
	}
	err := d.session.Destroy()
	d.session = nil
	if d.sessOpts != nil {
		d.sessOpts.Destroy()
		d.sessOpts = nil
	}
	return err
}

// Diacritize turns bare (or already vocalized) Arabic into fully-vocalized
// text. Diacritics are stripped first, so the result is canonical and
// idempotent; non-Arabic characters pass through unchanged.
//
//	Diacritize("محمد قرأ الكتاب") -> "مُحَمَّدٌ قَرَأَ الْكِتَابِ"
func (d *Diacritizer) Diacritize(text string) (string, error) {
	raw := text
	for len(raw) > 0 && (raw[0] == ' ' || raw[0] == '\t' || raw[0] == '\n' || raw[0] == '\r') {
		raw = raw[1:]
	}
	for len(raw) > 0 && (raw[len(raw)-1] == ' ' || raw[len(raw)-1] == '\t' || raw[len(raw)-1] == '\n' || raw[len(raw)-1] == '\r') {
		raw = raw[:len(raw)-1]
	}
	if raw == "" {
		return "", nil
	}
	base := StripDiacritics(raw)
	if base == "" {
		return raw, nil
	}
	labels, err := d.PredictLabels(base)
	if err != nil {
		return "", err
	}
	return Attach(base, labels), nil
}

// PredictLabels returns the predicted label id (0..NumLabels-1) for every
// character of text (diacritics stripped first). Empty input yields a nil
// slice.
func (d *Diacritizer) PredictLabels(text string) ([]int, error) {
	base := StripDiacritics(text)
	if base == "" {
		return nil, nil
	}
	runes := []rune(base)
	n := len(runes)

	ids := Encode(base, d.sym2ID, d.unkID)
	ids64 := make([]int64, len(ids))
	for i, v := range ids {
		ids64[i] = int64(v)
	}

	in, err := ort.NewTensor[int64](ort.NewShape(1, int64(n)), ids64)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInference, err)
	}
	defer in.Destroy()

	out, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(n), int64(d.numLabels)))
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInference, err)
	}
	defer out.Destroy()

	if err := d.session.Run([]ort.Value{in}, []ort.Value{out}); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInference, err)
	}

	logits := out.GetData()
	labels := make([]int, n)
	for i := 0; i < n; i++ {
		off := i * d.numLabels
		if off+d.numLabels > len(logits) {
			break
		}
		labels[i] = Argmax(logits[off : off+d.numLabels])
	}
	return labels, nil
}

// RepoID returns the Hugging Face repository the model came from ("<local>"
// for NewFromFiles).
func (d *Diacritizer) RepoID() string { return d.repoID }

// Revision returns the branch / tag / commit (or model file name) the model
// came from.
func (d *Diacritizer) Revision() string { return d.revision }

// NumLabels returns the number of distinct labels (13).
func (d *Diacritizer) NumLabels() int { return d.numLabels }

// MaxSeqLen returns the model's maximum sequence length (1024).
func (d *Diacritizer) MaxSeqLen() int { return d.maxSeqLen }

func (d *Diacritizer) String() string {
	return fmt.Sprintf("Diacritizer(repo_id=%q, revision=%q, num_labels=%d)",
		d.repoID, d.revision, d.numLabels)
}

// ensureEnvironment initializes the ONNX Runtime environment exactly once per
// process. The shared library path can be overridden with the
// ICHKIL_ONNXRUNTIME_LIB environment variable; otherwise the standard
// onnxruntime.so / onnxruntime.dll / libonnxruntime.dylib lookup is used.
var (
	envOnce sync.Once
	envErr  error
)

func ensureEnvironment() error {
	envOnce.Do(func() {
		if p := os.Getenv("ICHKIL_ONNXRUNTIME_LIB"); p != "" {
			ort.SetSharedLibraryPath(p)
		}
		if err := ort.InitializeEnvironment(); err != nil {
			envErr = fmt.Errorf("%w: failed to initialize onnxruntime: %v", ErrModelLoad, err)
			return
		}
		// Telemetry is off by default; ignore failure (non-fatal).
		_ = ort.DisableTelemetry()
	})
	return envErr
}

// RuntimeVersion returns the version of the loaded ONNX Runtime shared
// library (empty string when it is not available).
func RuntimeVersion() string {
	if err := ensureEnvironment(); err != nil {
		return ""
	}
	return ort.GetVersion()
}

// ---------------------------------------------------------------------------
// Module-level convenience API (one shared default Diacritizer).
// ---------------------------------------------------------------------------

var (
	defaultMu     sync.Mutex
	defaultDiacer *Diacritizer
)

func defaultDiacritize(ctx context.Context) (*Diacritizer, error) {
	defaultMu.Lock()
	defer defaultMu.Unlock()
	if defaultDiacer != nil {
		return defaultDiacer, nil
	}
	d, err := New(ctx)
	if err != nil {
		return nil, err
	}
	defaultDiacer = d
	return d, nil
}

// Diacritize vocalizes bare Arabic text using a shared default Diacritizer
// (created lazily on first use — it downloads the model once, then works
// offline).
func Diacritize(ctx context.Context, text string) (string, error) {
	d, err := defaultDiacritize(ctx)
	if err != nil {
		return "", err
	}
	return d.Diacritize(text)
}

// PredictLabels returns per-character labels for text using a shared default
// Diacritizer.
func PredictLabels(ctx context.Context, text string) ([]int, error) {
	d, err := defaultDiacritize(ctx)
	if err != nil {
		return nil, err
	}
	return d.PredictLabels(text)
}

// SetDefault installs d as the shared default Diacritizer used by Diacritize
// and PredictLabels.
func SetDefault(d *Diacritizer) {
	defaultMu.Lock()
	defer defaultMu.Unlock()
	if defaultDiacer != nil {
		_ = defaultDiacer.Close()
	}
	defaultDiacer = d
}

// ResetDefault discards the shared default Diacritizer (if any), forcing a
// fresh model resolution + session on the next convenience call.
func ResetDefault() {
	defaultMu.Lock()
	defer defaultMu.Unlock()
	if defaultDiacer != nil {
		_ = defaultDiacer.Close()
		defaultDiacer = nil
	}
}
