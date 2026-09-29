package ichkil

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Defaults for model resolution.
const (
	// DefaultRepoID is the Hugging Face model repository.
	DefaultRepoID = "ichkil/ichkil"
	// DefaultRevision is the branch fetched by default.
	DefaultRevision = "main"
)

// Config is the runtime configuration loaded from the model's config.json.
type Config struct {
	// Sym2ID maps every vocabulary symbol to its token id.
	Sym2ID map[string]int `json:"sym2id"`
	// NumTags is the number of output labels (13).
	NumTags int `json:"num_tags"`
	// MaxSeqLen is the maximum sequence length of the model (1024).
	MaxSeqLen int `json:"max_seq_len"`
	// SHA256 is the optional expected digest of model.onnx (fallback when the
	// repository has no SHA256SUMS file).
	SHA256 string `json:"sha256"`
}

// ModelBundle is a resolved, verified local model artifact plus its config.
type ModelBundle struct {
	// OnnxPath is the absolute path to the local model.onnx.
	OnnxPath string
	// Config is the parsed runtime configuration.
	Config Config
	// RepoID is the Hugging Face repository the bundle came from.
	RepoID string
	// Revision is the branch / tag / commit the bundle was fetched from.
	Revision string
}

var httpClient = &http.Client{Timeout: 10 * time.Minute}

func hfBaseURL() string {
	if u := os.Getenv("HF_ENDPOINT"); u != "" {
		return strings.TrimSuffix(u, "/")
	}
	return "https://huggingface.co"
}

func defaultCacheDir() string {
	if d := os.Getenv("ICHKIL_CACHE_DIR"); d != "" {
		return d
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		return filepath.Join(home, ".cache", "ichkil")
	}
	return filepath.Join(os.TempDir(), "ichkil")
}

// LoadConfig loads and minimally validates a config.json.
func LoadConfig(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("%w: %v", ErrModelLoad, err)
	}
	var raw struct {
		Sym2ID    *map[string]int `json:"sym2id"`
		NumTags   *int            `json:"num_tags"`
		MaxSeqLen *int            `json:"max_seq_len"`
		SHA256    *string         `json:"sha256"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return Config{}, fmt.Errorf("%w: invalid config.json %s: %v", ErrModelLoad, path, err)
	}
	if raw.Sym2ID == nil || raw.NumTags == nil {
		missing := ""
		if raw.Sym2ID == nil {
			missing = "sym2id"
		}
		if raw.NumTags == nil {
			missing = "num_tags"
		}
		return Config{}, fmt.Errorf("%w: config.json is missing required key '%s' (from %s)", ErrModelLoad, missing, path)
	}
	cfg := Config{
		Sym2ID:    *raw.Sym2ID,
		NumTags:   *raw.NumTags,
		MaxSeqLen: 0,
		SHA256:    "",
	}
	if raw.MaxSeqLen != nil {
		cfg.MaxSeqLen = *raw.MaxSeqLen
	}
	if raw.SHA256 != nil {
		cfg.SHA256 = *raw.SHA256
	}
	return cfg, nil
}

// SHA256OfFile returns the hex SHA-256 digest of the file at path.
func SHA256OfFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// VerifyChecksum returns an error wrapping ErrChecksumMismatch if the file's
// SHA-256 differs from expected (compared case-insensitively).
func VerifyChecksum(path, expected string) error {
	actual, err := SHA256OfFile(path)
	if err != nil {
		return err
	}
	if !strings.EqualFold(actual, strings.TrimSpace(expected)) {
		return fmt.Errorf("%w: %s: expected %s, got %s (delete the cached file or disable checksum verification)",
			ErrChecksumMismatch, path, strings.ToLower(strings.TrimSpace(expected)), actual)
	}
	return nil
}

// parseSHA256Sums extracts the expected SHA-256 for filename from a
// SHA256SUMS document ("" when absent).
func parseSHA256Sums(text, filename string) string {
	scanner := bufio.NewScanner(strings.NewReader(text))
	for scanner.Scan() {
		parts := strings.Fields(scanner.Text())
		if len(parts) >= 2 && strings.TrimPrefix(parts[len(parts)-1], "*") == filename {
			return parts[0]
		}
	}
	return ""
}

// ResolveModel downloads (or reuses the cached) model files from Hugging
// Face: model.onnx, config.json and — when checksum verification is enabled —
// SHA256SUMS. Everything is cached under $ICHKIL_CACHE_DIR (or ~/.cache/ichkil
// by default), so only the first call per revision touches the network.
func ResolveModel(ctx context.Context, cfg *config) (ModelBundle, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	repoDir := filepath.Join(cfg.cacheDir, "models--"+strings.ReplaceAll(cfg.repoID, "/", "--"))
	if err := os.MkdirAll(repoDir, 0o755); err != nil {
		return ModelBundle{}, fmt.Errorf("%w: cannot create cache dir %s: %v", ErrModelLoad, repoDir, err)
	}

	onnxPath := filepath.Join(repoDir, cfg.revision, "model.onnx")
	configPath := filepath.Join(repoDir, cfg.revision, "config.json")
	sumsPath := filepath.Join(repoDir, cfg.revision, "SHA256SUMS")

	onnxPath, err := ensureFile(ctx, cfg, "model.onnx", onnxPath, false)
	if err != nil {
		return ModelBundle{}, err
	}
	configPath, err = ensureFile(ctx, cfg, "config.json", configPath, false)
	if err != nil {
		return ModelBundle{}, err
	}

	config, err := LoadConfig(configPath)
	if err != nil {
		return ModelBundle{}, err
	}

	if cfg.verifyChecksum {
		expected := ""
		if sumsPath, err := ensureFile(ctx, cfg, "SHA256SUMS", sumsPath, true); err == nil {
			if data, rerr := os.ReadFile(sumsPath); rerr == nil {
				expected = parseSHA256Sums(string(data), "model.onnx")
			}
		}
		if expected == "" {
			expected = config.SHA256
		}
		if expected == "" {
			return ModelBundle{}, fmt.Errorf(
				"%w: cannot verify model checksum: the repository has neither SHA256SUMS nor config.sha256 (pass WithoutChecksumVerification() to bypass)",
				ErrModelLoad)
		}
		if err := VerifyChecksum(onnxPath, expected); err != nil {
			return ModelBundle{}, err
		}
	}

	return ModelBundle{
		OnnxPath: onnxPath,
		Config:   config,
		RepoID:   cfg.repoID,
		Revision: cfg.revision,
	}, nil
}

// ensureFile returns the destination path for filename, downloading it from
// the Hugging Face repository when the cached copy is missing. When optional
// is true, download failures are returned as-is so the caller may ignore them
// (used for the optional SHA256SUMS file); required files get errors wrapped
// in ErrModelLoad.
func ensureFile(ctx context.Context, cfg *config, filename, dest string, optional bool) (string, error) {
	if fi, err := os.Stat(dest); err == nil && fi.Size() > 0 {
		return dest, nil
	}
	err := downloadFile(ctx, cfg, filename, dest)
	if err != nil {
		if !optional {
			return "", fmt.Errorf("%w: cannot fetch %s: %v", ErrModelLoad, filename, err)
		}
		return "", err
	}
	return dest, nil
}

// downloadFile fetches one file from https://huggingface.co/<repo>/resolve/<rev>/<file>
// into dest (atomically, via a temporary .part file).
func downloadFile(ctx context.Context, cfg *config, filename, dest string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	u := fmt.Sprintf("%s/%s/resolve/%s/%s",
		cfg.hfBaseURL,
		cfg.repoID,
		url.PathEscape(cfg.revision),
		url.PathEscape(filename))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "ichkil-go/"+Version)
	if token := cfg.token; token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status %s", resp.Status)
	}

	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	tmp := dest + ".part"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, resp.Body); err != nil {
		out.Close()
		os.Remove(tmp)
		return err
	}
	if err := out.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, dest); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}
