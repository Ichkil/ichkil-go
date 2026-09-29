package ichkil

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
)

// Integration tests for the ONNX-backed Diacritizer.
//
// They download the model from Hugging Face on the first run (≈5 MB, cached
// under ~/.cache/ichkil afterwards) and require the onnxruntime shared
// library — point ICHKIL_ONNXRUNTIME_LIB at it, e.g.:
//
//	ICHKIL_ONNXRUNTIME_LIB=$(python -c "import onnxruntime; print(onnxruntime.get_library_path())") \
//	    go test -v ./...
//
// When the runtime library is unavailable the tests are skipped, so
// `go test ./...` stays green on machines without ONNX Runtime.

var (
	testDiacerOnce sync.Once
	testDiacer     *Diacritizer
	testDiacerErr  error
)

func skipIfNoRuntime(t *testing.T) {
	t.Helper()
	if err := ensureEnvironment(); err != nil {
		t.Skipf("onnxruntime not available (set ICHKIL_ONNXRUNTIME_LIB to the onnxruntime shared library): %v", err)
	}
}

func testDiacritizer(t *testing.T) *Diacritizer {
	t.Helper()
	skipIfNoRuntime(t)
	testDiacerOnce.Do(func() {
		testDiacer, testDiacerErr = New(context.Background())
	})
	if testDiacerErr != nil {
		t.Fatalf("New() = %v", testDiacerErr)
	}
	return testDiacer
}

func TestGoldenVectors(t *testing.T) {
	d := testDiacritizer(t)
	for _, vec := range loadGoldenVectors(t) {
		got, err := d.Diacritize(vec.Input)
		if err != nil {
			t.Fatalf("%s (%s): %v", vec.ID, vec.Category, err)
		}
		if got != vec.ExpectedText {
			t.Errorf("%s (%s)\ninput    : %s\nexpected : %s\ngot      : %s",
				vec.ID, vec.Category, vec.Input, vec.ExpectedText, got)
		}
	}
}

func TestMaxSeqLen1024(t *testing.T) {
	d := testDiacritizer(t)
	var long *goldenVector
	for _, vec := range loadGoldenVectors(t) {
		if vec.Category == "long" {
			long = &vec
		}
	}
	if long == nil {
		t.Fatal("golden set is missing the 'long' vector")
	}
	if len([]rune(long.Input)) != d.MaxSeqLen() {
		t.Fatalf("long vector length = %d, MaxSeqLen() = %d", len([]rune(long.Input)), d.MaxSeqLen())
	}
	got, err := d.Diacritize(long.Input)
	if err != nil {
		t.Fatal(err)
	}
	if got != long.ExpectedText {
		t.Errorf("long vector:\nexpected: %s\ngot     : %s", long.ExpectedText, got)
	}
}

func TestEmptyAndWhitespace(t *testing.T) {
	d := testDiacritizer(t)
	for _, in := range []string{"", "   ", "\t\n", " \t \n "} {
		got, err := d.Diacritize(in)
		if err != nil {
			t.Fatalf("Diacritize(%q) = %v", in, err)
		}
		if got != "" {
			t.Errorf("Diacritize(%q) = %q, want \"\"", in, got)
		}
	}
	labels, err := d.PredictLabels("")
	if err != nil {
		t.Fatal(err)
	}
	if labels != nil {
		t.Errorf("PredictLabels(\"\") = %v, want nil", labels)
	}
	// whitespace is not empty: one label per character, like any other text
	labels, err = d.PredictLabels("   ")
	if err != nil {
		t.Fatal(err)
	}
	if len(labels) != 3 {
		t.Errorf("len(PredictLabels(\"   \")) = %d, want 3", len(labels))
	}
	for i, lab := range labels {
		if lab < 0 || lab >= 13 {
			t.Fatalf("labels[%d] = %d out of range [0,13)", i, lab)
		}
	}
}

func TestNonArabicPassthrough(t *testing.T) {
	d := testDiacritizer(t)
	cases := []string{
		"Hello 123!",
		"café résumé",
		"abc def 42",
	}
	for _, in := range cases {
		got, err := d.Diacritize(in)
		if err != nil {
			t.Fatalf("Diacritize(%q) = %v", in, err)
		}
		if got != in {
			t.Errorf("Diacritize(%q) = %q, want passthrough", in, got)
		}
	}
}

func TestMixedArabicLatin(t *testing.T) {
	d := testDiacritizer(t)
	got, err := d.Diacritize("محمد said: الكتاب!")
	if err != nil {
		t.Fatal(err)
	}
	// Latin / punctuation must pass through unchanged.
	for _, frag := range []string{"said:", "!"} {
		if !strings.Contains(got, frag) {
			t.Errorf("result %q is missing passthrough fragment %q", got, frag)
		}
	}
}

// TestIdempotency mirrors Python's test_idempotent_on_vocalized_text: these
// already-vocalized strings are verified fixed points of the model (the same
// strings the Python suite asserts, so the Go runtime is held to the exact
// same contract).
func TestIdempotency(t *testing.T) {
	d := testDiacritizer(t)
	for _, in := range []string{
		"مُحَمَّدٌ قَرَأَ الْكِتَابَ فِي الْمَدْرَسَةِ",
		"الْعِلْمِ نُورٌ وَالْجَهْلُ ظُلْمَةُ",
		"مَدْرَسَةٌ",
	} {
		got, err := d.Diacritize(in)
		if err != nil {
			t.Fatalf("Diacritize(%q) = %v", in, err)
		}
		if got != in {
			t.Errorf("Diacritize(%q) = %q, want idempotent %q", in, got, in)
		}
	}
}

// TestNormalization mirrors Python's test_input_with_diacritics_is_normalized:
// stripping then re-attaching yields the canonical golden form.
func TestNormalization(t *testing.T) {
	d := testDiacritizer(t)
	for _, in := range []string{"مُحَمَّدٌ", "كِتَابِ"} {
		got, err := d.Diacritize(in)
		if err != nil {
			t.Fatalf("Diacritize(%q) = %v", in, err)
		}
		if got != in {
			t.Errorf("Diacritize(%q) = %q, want canonical %q", in, got, in)
		}
	}
}

func TestPredictLabelsShapeAndRange(t *testing.T) {
	d := testDiacritizer(t)
	text := "العلم نور"
	labels, err := d.PredictLabels(text)
	if err != nil {
		t.Fatal(err)
	}
	if want := len([]rune(text)); len(labels) != want {
		t.Fatalf("len(labels) = %d, want %d (per character)", len(labels), want)
	}
	for i, lab := range labels {
		if lab < 0 || lab >= 13 {
			t.Fatalf("labels[%d] = %d out of range [0,13)", i, lab)
		}
	}
	// non-letter characters get "no diacritic" (0 or 1); attach() ignores them anyway
	spaceIdx := 0
	for i, r := range []rune(text) {
		if r == ' ' {
			spaceIdx = i
			break
		}
	}
	if labels[spaceIdx] != 0 && labels[spaceIdx] != 1 {
		t.Errorf("labels[space] = %d, want 0 or 1", labels[spaceIdx])
	}
}

func TestNewFromFilesParity(t *testing.T) {
	d := testDiacritizer(t)
	cfg := defaultConfig()
	bundle, err := ResolveModel(context.Background(), &cfg)
	if err != nil {
		t.Fatalf("ResolveModel = %v", err)
	}
	offline, err := NewFromFiles(bundle.OnnxPath)
	if err != nil {
		t.Fatalf("NewFromFiles = %v", err)
	}
	defer offline.Close()

	if offline.RepoID() != "<local>" {
		t.Errorf("RepoID = %q, want <local>", offline.RepoID())
	}
	if offline.NumLabels() != d.NumLabels() || offline.MaxSeqLen() != d.MaxSeqLen() {
		t.Errorf("offline = (%d, %d), online = (%d, %d)",
			offline.NumLabels(), offline.MaxSeqLen(), d.NumLabels(), d.MaxSeqLen())
	}
	for _, vec := range loadGoldenVectors(t) {
		got, err := offline.Diacritize(vec.Input)
		if err != nil {
			t.Fatalf("%s: %v", vec.ID, err)
		}
		want, err := d.Diacritize(vec.Input)
		if err != nil {
			t.Fatalf("%s (online): %v", vec.ID, err)
		}
		if got != want {
			t.Errorf("%s: offline = %q, online = %q", vec.ID, got, want)
		}
	}
}

func TestNewFromFilesMissingModel(t *testing.T) {
	_, err := NewFromFiles(t.TempDir() + "/absent.onnx")
	if !errors.Is(err, ErrModelLoad) {
		t.Fatalf("err = %v, want ErrModelLoad", err)
	}
}

func TestVersionAndPublicAPI(t *testing.T) {
	if Version != "1.0.0" {
		t.Errorf("Version = %q, want 1.0.0", Version)
	}
	if NumLabels != 13 {
		t.Errorf("NumLabels = %d, want 13", NumLabels)
	}
	// the module-level convenience API exists and delegates to the same code
	_ = Diacritize
	_ = PredictLabels
	_ = SetDefault
	_ = ResetDefault
	_ = WithRepoID
	_ = WithRevision
	_ = WithCacheDir
	_ = WithToken
	_ = WithNumThreads
	_ = WithoutChecksumVerification
	_ = WithBaseURL
	_ = WithConfigPath
	_ = RuntimeVersion
}

func TestString(t *testing.T) {
	d := testDiacritizer(t)
	s := d.String()
	if !strings.Contains(s, "ichkil/ichkil") {
		t.Errorf("String() = %q, want it to mention the repo", s)
	}
	if !strings.Contains(s, "13") {
		t.Errorf("String() = %q, want it to mention the label count", s)
	}
}

func TestCloseIdempotent(t *testing.T) {
	skipIfNoRuntime(t)
	cfg := defaultConfig()
	bundle, err := ResolveModel(context.Background(), &cfg)
	if err != nil {
		t.Fatalf("ResolveModel = %v", err)
	}
	d, err := NewFromFiles(bundle.OnnxPath)
	if err != nil {
		t.Fatalf("NewFromFiles = %v", err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("Close() = %v", err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("Close() again = %v, want nil (idempotent)", err)
	}
}
