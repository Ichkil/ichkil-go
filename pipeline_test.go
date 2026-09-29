package ichkil

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// Pure-pipeline tests — no model, no network, no third-party dependencies.
//
// These lock in the text contract (label table, strip/encode/decode/attach)
// shared by every ichkil client runtime. The golden vectors in
// testdata/golden.json are a verbatim copy of the core repository's
// golden/vectors.json (cross-runtime contract).

// goldenVector is one entry of the cross-runtime golden set, a verbatim
// copy of the core repository's golden/vectors.json.
type goldenVector struct {
	ID           string `json:"id"`
	Category     string `json:"category"`
	Input        string `json:"input"`
	Labels       []int  `json:"labels"`
	ExpectedText string `json:"expected_text"`
}

func loadGoldenVectors(t *testing.T) []goldenVector {
	t.Helper()
	data, err := os.ReadFile("testdata/golden.json")
	if err != nil {
		t.Fatalf("read golden vectors: %v", err)
	}
	var doc struct {
		Vectors []goldenVector `json:"vectors"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatalf("parse golden vectors: %v", err)
	}
	return doc.Vectors
}

func TestLabelTableCoversAll13Labels(t *testing.T) {
	if NumLabels != 13 {
		t.Fatalf("NumLabels = %d, want 13", NumLabels)
	}
	if len(labelToCombo) != NumLabels {
		t.Fatalf("len(labelToCombo) = %d, want %d", len(labelToCombo), NumLabels)
	}
	if labelToCombo[0] != "" || labelToCombo[1] != "" {
		t.Fatalf("labels 0/1 must carry no diacritic: %q %q", labelToCombo[0], labelToCombo[1])
	}
	for i := 2; i < NumLabels; i++ {
		if labelToCombo[i] == "" {
			t.Fatalf("label %d must carry a diacritic combination", i)
		}
	}
	// shadda+haraka combos contain the shadda mark
	for _, i := range []int{7, 8, 9} {
		if !strings.Contains(labelToCombo[i], "\u0651") {
			t.Errorf("label %d must contain shadda U+0651: %q", i, labelToCombo[i])
		}
	}
	// tanwin combos
	if got := labelToCombo[10]; got != "\u064b" {
		t.Errorf("label 10 = %q, want U+064B", got)
	}
	if got := labelToCombo[11]; got != "\u064c" {
		t.Errorf("label 11 = %q, want U+064C", got)
	}
	if got := labelToCombo[12]; got != "\u064d" {
		t.Errorf("label 12 = %q, want U+064D", got)
	}
}

func TestLabelToComboRoundtrip(t *testing.T) {
	for _, label := range []int{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} {
		combo := LabelToCombo(label)
		if combo == "" {
			t.Errorf("LabelToCombo(%d) = \"\", want non-empty", label)
			continue
		}
		if got := ComboToLabel(combo); got != label {
			t.Errorf("ComboToLabel(%q) = %d, want %d", combo, got, label)
		}
	}
}

func TestComboToLabelUnknownDefaultsToZero(t *testing.T) {
	if got := ComboToLabel(""); got != LabelNonLetter {
		t.Errorf("ComboToLabel(\"\") = %d, want %d", got, LabelNonLetter)
	}
	if got := ComboToLabel("\u0670"); got != LabelNonLetter {
		t.Errorf("ComboToLabel(U+0670) = %d, want %d", got, LabelNonLetter)
	}
}

func TestLabelToComboOutOfRange(t *testing.T) {
	if got := LabelToCombo(-1); got != "" {
		t.Errorf("LabelToCombo(-1) = %q, want \"\"", got)
	}
	if got := LabelToCombo(NumLabels); got != "" {
		t.Errorf("LabelToCombo(%d) = %q, want \"\"", NumLabels, got)
	}
}

func TestIsArabicLetter(t *testing.T) {
	cases := []struct {
		r        rune
		expected bool
	}{
		{'\u0627', true}, // alif
		{'\u0628', true}, // ba
		{'\u0649', true}, // yeh
		{'\u064a', true}, // yeh (final range)
		{'a', false},
		{'1', false},
		{' ', false},
		{'\u0640', true},  // tatweel is inside the reference letter range
		{'\u0650', false}, // kasra is a diacritic, not a letter
	}
	for _, c := range cases {
		if got := IsArabicLetter(c.r); got != c.expected {
			t.Errorf("IsArabicLetter(%q) = %v, want %v", c.r, got, c.expected)
		}
	}
}

func TestStripDiacriticsRemovesHarakatAndTatweel(t *testing.T) {
	vocalized := "مُحَمَّدٌ قَرَأَ الْكِتَابَ"
	if got := StripDiacritics(vocalized); got != "محمد قرأ الكتاب" {
		t.Errorf("StripDiacritics(%q) = %q, want %q", vocalized, got, "محمد قرأ الكتاب")
	}
	if got := StripDiacritics("مـحـمـد"); got != "محمد" {
		t.Errorf("StripDiacritics(tatweel) = %q, want %q", got, "محمد")
	}
}

func TestStripDiacriticsKeepsLettersAndPunctuation(t *testing.T) {
	if got := StripDiacritics("ا،b"); got != "ا،b" {
		t.Errorf("StripDiacritics = %q, want %q", got, "ا،b")
	}
}

func TestEncodeFallbackToUnk(t *testing.T) {
	sym2ID := map[string]int{"ا": 2, "ب": 3, " ": 7}
	ids := Encode("اب?", sym2ID, 1)
	want := []int{2, 3, 1}
	if len(ids) != len(want) {
		t.Fatalf("Encode len = %d, want %d", len(ids), len(want))
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("Encode[%d] = %d, want %d", i, ids[i], want[i])
		}
	}
}

func TestArgmaxFirstWinsTies(t *testing.T) {
	if got := Argmax([]float32{1, 3, 3, 2}); got != 1 {
		t.Errorf("Argmax = %d, want 1 (first wins ties)", got)
	}
	if got := Argmax([]float32{5}); got != 0 {
		t.Errorf("Argmax single = %d, want 0", got)
	}
	if got := Argmax([]float32{-2, -5, -1, -9}); got != 2 {
		t.Errorf("Argmax negatives = %d, want 2", got)
	}
}

func TestAttachShortLabelsLeavesRestPlain(t *testing.T) {
	if got := Attach("اب", []int{2, 3}); got != "اَ"+"بُ" {
		t.Errorf("Attach(اب, [2,3]) = %q, want %q", got, "اَ"+"بُ")
	}
	if got := Attach("اب", []int{2}); got != "اَ"+"ب" {
		t.Errorf("Attach short = %q, want %q", got, "اَ"+"ب")
	}
	if got := Attach("ا ب", []int{1, 0, 2}); got != "ا بَ" {
		t.Errorf("Attach non-letter = %q, want %q", got, "ا بَ")
	}
}

func TestDecodeFromFlatLogits(t *testing.T) {
	// one char "ب": 13 logits, max at index 4 (kasra)
	logits := make([]float32, 13)
	logits[4] = 9
	if got := Decode("ب", logits, NumLabels); got != "بِ" {
		t.Errorf("Decode = %q, want %q", got, "بِ")
	}
	// two chars; second all zeros -> argmax 0 -> no decoration (non-letter)
	logits2 := make([]float32, 26)
	logits2[2] = 5 // first char: fatha
	if got := Decode("ب ", logits2, NumLabels); got != "بَ " {
		t.Errorf("Decode(ب , ...) = %q, want %q", got, "بَ ")
	}
}

func TestGoldenVectorsAttach(t *testing.T) {
	// Pure contract: attach(input, labels) must equal expected_text for every
	// cross-runtime golden vector.
	for _, vec := range loadGoldenVectors(t) {
		if got := Attach(vec.Input, vec.Labels); got != vec.ExpectedText {
			t.Errorf("%s (%s)\ninput    : %s\nexpected : %s\ngot      : %s",
				vec.ID, vec.Category, vec.Input, vec.ExpectedText, got)
		}
	}
}
