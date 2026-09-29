// Command basic demonstrates the minimal ichkil usage.
//
//	go run ./examples/basic
//
// On first run it downloads the ONNX model from Hugging Face (≈5 MB) and
// caches it under ~/.cache/ichkil; afterwards everything works offline.
package main

import (
	"context"
	"fmt"
	"log"

	ichkil "github.com/Ichkil/ichkil-go"
)

func main() {
	ctx := context.Background()

	// One-liner API — a shared default Diacritizer is created lazily.
	got, err := ichkil.Diacritize(ctx, "محمد قرأ الكتاب")
	if err != nil {
		log.Fatalf("diacritize: %v", err)
	}
	fmt.Println(got) // مُحَمَّدٌ قَرَأَ الْكِتَابَ

	// Explicit control.
	d, err := ichkil.New(ctx) // default repository: ichkil/ichkil@main
	if err != nil {
		log.Fatalf("new diacritizer: %v", err)
	}
	defer d.Close()

	fmt.Println(d.Diacritize("الكتاب على الطاولة"))

	// Per-character labels (0..12).
	labels, err := d.PredictLabels("العلم نور")
	if err != nil {
		log.Fatalf("predict labels: %v", err)
	}
	fmt.Println("labels:", labels)
}
