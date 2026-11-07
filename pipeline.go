package ichkil

import (
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// Pure text pipeline for Arabic Tashkeel — no model, no third-party imports.
//
// The model emits exactly one label per input character:
//
//	0        non-letter token (space / punctuation / digit / Latin) or padding
//	1        letter carries no diacritic
//	2 .. 15  the harakat / shadda / tanwin combinations
//	         (13..15 = shadda + tanwin, added in model v0.3)
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
	NumLabels = 16
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
	"\u0651\u064b", // 13 shadda + tanwin fatha
	"\u0651\u064c", // 14 shadda + tanwin damma
	"\u0651\u064d", // 15 shadda + tanwin kasra
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

// IsArabicLetter reports whether r is an Arabic letter
// (U+0621..U+063A, U+0641..U+064A; tatweel is not a letter).
func IsArabicLetter(r rune) bool {
	return (r >= '\u0621' && r <= '\u063A') || (r >= '\u0641' && r <= '\u064A')
}

// charMap maps presentation variants to the base Arabic letters the model knows.
var charMap = map[rune]rune{
	'\u0671': '\u0627', // alef wasla -> alef
	'\u06CC': '\u064A', // farsi yeh -> yeh
	'\u06A9': '\u0643', // keheh -> kaf
	'\u06C1': '\u0647', // heh goal -> heh
	'\u00A0': ' ',      // no-break space
}

// NormalizeText returns the canonical form used by the model (mirror of the
// core normalize_text): Unicode NFC, presentation variants mapped to base
// Arabic letters, zero-width characters removed, whitespace runs collapsed to
// one space, trimmed. Diacritics are kept (see StripDiacritics).
func NormalizeText(text string) string {
	text = norm.NFC.String(text)
	var b strings.Builder
	b.Grow(len(text))
	space := false
	for _, r := range text {
		if m, ok := charMap[r]; ok {
			r = m
		}
		if (r >= '\u200B' && r <= '\u200F') || r == '\uFEFF' {
			continue
		}
		if unicode.IsSpace(r) {
			space = true
			continue
		}
		if space && b.Len() > 0 {
			b.WriteByte(' ')
		}
		space = false
		b.WriteRune(r)
	}
	return b.String()
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

// StripDiacritics normalizes text (see NormalizeText), then removes all
// diacritics and tatweel. Letters, spaces and punctuation are preserved so the
// result stays aligned with the characters the model sees.
func StripDiacritics(text string) string {
	text = NormalizeText(text)
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
