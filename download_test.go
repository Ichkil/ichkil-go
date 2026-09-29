package ichkil

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Download / verification tests. The ResolveModel cases seed a fake local
// cache so they run fully offline (no network, no real model).

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func digestOf(content string) string {
	sum := sha256.Sum256([]byte(content))
	return hex.EncodeToString(sum[:])
}

func TestSHA256OfFile(t *testing.T) {
	p := filepath.Join(t.TempDir(), "f.bin")
	writeFile(t, p, "ichkil model bytes")
	want := digestOf("ichkil model bytes")
	got, err := SHA256OfFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("SHA256OfFile = %s, want %s", got, want)
	}
}

func TestVerifyChecksumMismatch(t *testing.T) {
	p := filepath.Join(t.TempDir(), "model.onnx")
	writeFile(t, p, "fake model bytes")
	err := VerifyChecksum(p, strings.Repeat("0", 64))
	if !errors.Is(err, ErrChecksumMismatch) {
		t.Fatalf("err = %v, want ErrChecksumMismatch", err)
	}
}

func TestVerifyChecksumAcceptsRealDigestCaseInsensitive(t *testing.T) {
	p := filepath.Join(t.TempDir(), "model.onnx")
	writeFile(t, p, "fake model bytes")
	real := digestOf("fake model bytes")
	if err := VerifyChecksum(p, real); err != nil {
		t.Fatalf("VerifyChecksum(real) = %v, want nil", err)
	}
	if err := VerifyChecksum(p, strings.ToUpper(real)); err != nil {
		t.Fatalf("VerifyChecksum(UPPER real) = %v, want nil", err)
	}
	if err := VerifyChecksum(p, "  "+real+"  "); err != nil {
		t.Fatalf("VerifyChecksum(padded real) = %v, want nil", err)
	}
}

func TestVerifyChecksumMissingFile(t *testing.T) {
	err := VerifyChecksum(filepath.Join(t.TempDir(), "absent.onnx"), strings.Repeat("0", 64))
	if err == nil {
		t.Fatal("expected an error for a missing file")
	}
}

func TestParseSHA256Sums(t *testing.T) {
	doc := "9055816b214346b0a8044a4a8deceb36e818dffd0b04699715ea23883bf54744  model.onnx\n" +
		"deadbeef  other.bin\n" +
		"cafe  *another.bin\n"
	if got := parseSHA256Sums(doc, "model.onnx"); got != "9055816b214346b0a8044a4a8deceb36e818dffd0b04699715ea23883bf54744" {
		t.Fatalf("parseSHA256Sums = %q", got)
	}
	if got := parseSHA256Sums(doc, "other.bin"); got != "deadbeef" {
		t.Fatalf("parseSHA256Sums(*other) = %q", got)
	}
	if got := parseSHA256Sums(doc, "missing.bin"); got != "" {
		t.Fatalf("parseSHA256Sums(missing) = %q, want \"\"", got)
	}
}

func TestLoadConfigValid(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "config.json")
	writeFile(t, p, `{
	  "sym2id": {"<unk>": 1, "\u0627": 2},
	  "num_tags": 13,
	  "max_seq_len": 1024,
	  "sha256": "9055816b214346b0a8044a4a8deceb36e818dffd0b04699715ea23883bf54744"
	}`)
	cfg, err := LoadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.NumTags != 13 || cfg.MaxSeqLen != 1024 {
		t.Fatalf("cfg = %+v", cfg)
	}
	if cfg.Sym2ID["\u0627"] != 2 || cfg.Sym2ID["<unk>"] != 1 {
		t.Fatalf("sym2id = %v", cfg.Sym2ID)
	}
	if cfg.SHA256 == "" {
		t.Fatal("sha256 missing")
	}
}

func TestLoadConfigMissingRequiredKeys(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "config.json")
	writeFile(t, p, `{"max_seq_len": 1024}`)
	_, err := LoadConfig(p)
	if !errors.Is(err, ErrModelLoad) {
		t.Fatalf("err = %v, want ErrModelLoad", err)
	}
}

func TestLoadConfigMissingFile(t *testing.T) {
	_, err := LoadConfig(filepath.Join(t.TempDir(), "absent.json"))
	if !errors.Is(err, ErrModelLoad) {
		t.Fatalf("err = %v, want ErrModelLoad", err)
	}
}

// seedCache fills cacheDir/<revision>/ with model.onnx, config.json and
// SHA256SUMS (all fake content).
func seedCache(t *testing.T, cacheDir, revision, modelContent, sumsDigest string) {
	t.Helper()
	base := filepath.Join(cacheDir, "models--ichkil--ichkil", revision)
	writeFile(t, filepath.Join(base, "model.onnx"), modelContent)
	writeFile(t, filepath.Join(base, "config.json"),
		`{"sym2id": {"<unk>": 1, "\u0627": 2}, "num_tags": 13, "max_seq_len": 1024}`)
	if sumsDigest != "" {
		writeFile(t, filepath.Join(base, "SHA256SUMS"), sumsDigest+"  model.onnx\n")
	}
}

func TestResolveModelOfflineChecksumMismatch(t *testing.T) {
	cacheDir := t.TempDir()
	model := "fake model bytes"
	// SHA256SUMS carries a digest that does NOT match the cached model.
	seedCache(t, cacheDir, "main", model, strings.Repeat("0", 64))

	cfg := defaultConfig()
	cfg.cacheDir = cacheDir
	_, err := ResolveModel(context.Background(), &cfg)
	if !errors.Is(err, ErrChecksumMismatch) {
		t.Fatalf("err = %v, want ErrChecksumMismatch", err)
	}
}

func TestResolveModelOfflineChecksumPass(t *testing.T) {
	cacheDir := t.TempDir()
	model := "fake model bytes"
	seedCache(t, cacheDir, "main", model, digestOf(model))

	cfg := defaultConfig()
	cfg.cacheDir = cacheDir
	bundle, err := ResolveModel(context.Background(), &cfg)
	if err != nil {
		t.Fatalf("ResolveModel = %v, want success", err)
	}
	if bundle.RepoID != "ichkil/ichkil" || bundle.Revision != "main" {
		t.Fatalf("bundle = %+v", bundle)
	}
	if _, err := os.Stat(bundle.OnnxPath); err != nil {
		t.Fatalf("model file missing: %v", err)
	}
	if bundle.Config.NumTags != 13 {
		t.Fatalf("NumTags = %d, want 13", bundle.Config.NumTags)
	}
}

func TestResolveModelFallsBackToConfigSHA256(t *testing.T) {
	cacheDir := t.TempDir()
	model := "fake model bytes"
	// A repo that does not exist on Hugging Face: the SHA256SUMS fetch is
	// guaranteed to fail, so the expected digest must come from config.json
	// sha256. Deterministic whether or not the network is reachable.
	repoID := "ichkil/ichkil-test-nonexistent"
	base := filepath.Join(cacheDir, "models--"+strings.ReplaceAll(repoID, "/", "--"), "main")
	writeFile(t, filepath.Join(base, "model.onnx"), model)
	writeFile(t, filepath.Join(base, "config.json"),
		`{"sym2id": {"<unk>": 1, "\u0627": 2}, "num_tags": 13, "sha256": "`+digestOf(model)+`"}`)

	cfg := defaultConfig()
	cfg.repoID = repoID
	cfg.cacheDir = cacheDir
	cfg.verifyChecksum = true
	bundle, err := ResolveModel(context.Background(), &cfg)
	if err != nil {
		t.Fatalf("ResolveModel = %v, want success (config.sha256 fallback)", err)
	}
	if bundle.Config.SHA256 != digestOf(model) {
		t.Fatalf("SHA256 = %q", bundle.Config.SHA256)
	}
}
