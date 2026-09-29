package ichkil

import "strings"

// Pure text pipeline for Arabic Tashkeel — no model, no third-party imports.
//
// The model emits exactly one label per input character:
//
//	0        non-letter token (space / punctuation) or padding
//	1        letter carries no diacritic
//	2 .. 12  the eleven harakat / shadda / tanwin combinations
//
// This file is the single source of truth shared by inference and the tests,
// and it mirrors the reference implementation in the core repository and the
// Python / JavaScript libraries used by the Hugging Face Space. Keep them in
// sync.

// Label constants.
const (
	// LabelNonLetter is the label for non-letter tokens (spaces, punctuation)
	// or padding.
	LabelNonLetter = 0
	// LabelNone is the label for a letter that carries no diacritic.
	LabelNone = 1
	// NumLabels is the number of distinct labels the model can emit.
	NumLabels = 13
)

// tatweel is the horizontal line, removed from input before encoding.
const tatweel = '\u0640'

// labelToCombo maps a label id to its canonical diacritic combination string.
var labelToCombo = [...]string{
	"",             // 0  non-letter token
	"",             // 1  letter carries no diacritic
	"\u064e",       // 2  fatha
	"\u064f",       // 3  damma
	"\u0650",       // 4  kasra
	"\u0652",       // 5  sukun
	"\u0651",       // 6  shadda
	"\u0651\u064e", // 7  shadda + fatha
	"\u0651\u064f", // 8  shadda + damma
	"\u0651\u0650", // 9  shadda + kasra
	"\u064b",       // 10 tanwin fatha
	"\u064c",       // 11 tanwin damma
	"\u064d",       // 12 tanwin kasra
}

// comboToLabel maps a diacritic combination to its label id.
var comboToLabel = func() map[string]int {
	m := make(map[string]int, len(labelToCombo)-2)
	for i, combo := range labelToCombo {
		if combo != "" {
			m[combo] = i
		}
	}
	return m
}()

// IsArabicLetter reports whether r is a single Arabic letter (U+0621..U+064A).
func IsArabicLetter(r rune) bool {
	return r >= '\u0621' && r <= '\u064A'
}

// isDiacritic reports whether r is an Arabic diacritic / harakat code point.
func isDiacritic(r rune) bool {
	switch {
	case r >= '\u0610' && r <= '\u061A':
		return true
	case r >= '\u064B' && r <= '\u065F':
		return true
	case r == '\u0670':
		return true
	case r >= '\u06D6' && r <= '\u06ED':
		return true
	}
	return false
}

// StripDiacritics returns text with all diacritics and tatweel removed.
// Letters, spaces and punctuation are preserved so the result stays aligned
// with the original character positions.
func StripDiacritics(text string) string {
	var b strings.Builder
	b.Grow(len(text))
	for _, r := range text {
		if r == tatweel || isDiacritic(r) {
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

// LabelToCombo maps a label id to its diacritic combination ("" for unknown
// ids).
func LabelToCombo(label int) string {
	if label >= 0 && label < NumLabels {
		return labelToCombo[label]
	}
	return ""
}

// ComboToLabel maps a diacritic combination (or "") to its label id (0 if
// unknown).
func ComboToLabel(combo string) int {
	if lab, ok := comboToLabel[combo]; ok {
		return lab
	}
	return LabelNonLetter
}

// Argmax returns the index of the maximum value (first one wins ties),
// matching ONNX Runtime.
func Argmax(values []float32) int {
	best := 0
	for i := 1; i < len(values); i++ {
		if values[i] > values[best] {
			best = i
		}
	}
	return best
}

// Encode maps each rune to its vocabulary id (out-of-vocabulary -> unkID).
func Encode(text string, sym2ID map[string]int, unkID int) []int {
	ids := make([]int, 0, len(text))
	for _, r := range text {
		id, ok := sym2ID[string(r)]
		if !ok {
			id = unkID
		}
		ids = append(ids, id)
	}
	return ids
}

// Attach appends predicted diacritics onto the bare letters of base.
// labels is one entry per character of base (a shorter slice leaves the
// remaining characters undecorated). Non-letter characters pass through
// unchanged.
func Attach(base string, labels []int) string {
	runes := []rune(base)
	var b strings.Builder
	b.Grow(len(runes) * 2)
	for i, r := range runes {
		b.WriteRune(r)
		if IsArabicLetter(r) && i < len(labels) {
			b.WriteString(LabelToCombo(labels[i]))
		}
	}
	return b.String()
}

// Decode attaches predicted diacritics onto base from flat logits.
// logits is the model output row for base in row-major order
// (len(logits) == len(base) * numLabels). Non-letter characters pass through
// unchanged.
func Decode(base string, logits []float32, numLabels int) string {
	runes := []rune(base)
	labels := make([]int, 0, len(runes))
	for i := range runes {
		off := i * numLabels
		if off+numLabels > len(logits) {
			break
		}
		labels = append(labels, Argmax(logits[off:off+numLabels]))
	}
	return Attach(base, labels)
}
