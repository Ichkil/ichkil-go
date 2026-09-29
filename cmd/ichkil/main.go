// Command ichkil is a small command-line interface to the ichkil Arabic
// Tashkeel (diacritization) engine.
//
// Usage:
//
//	ichkil "محمد قرأ الكتاب"
//	ichkil --json "الكتاب على الطاولة"
//	cat text.txt | ichkil
//	ichkil --model ./model.onnx "محمد"     # offline
//
// Exit codes: 0 on success, 1 on model/inference failure, 2 on usage error.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/Ichkil/ichkil-go"
)

func main() { os.Exit(run()) }

func run() int {
	fs := flag.NewFlagSet("ichkil", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	fs.Usage = func() {
		out := fs.Output()
		fmt.Fprintln(out, "ichkil — Arabic Tashkeel (diacritization): vocalizes bare Arabic text.")
		fmt.Fprintln(out, "")
		fmt.Fprintln(out, "Usage:")
		fmt.Fprintln(out, `  ichkil "محمد قرأ الكتاب"`)
		fmt.Fprintln(out, `  ichkil --json "الكتاب على الطاولة"`)
		fmt.Fprintln(out, "  cat text.txt | ichkil")
		fmt.Fprintln(out, `  ichkil --model ./model.onnx "محمد"     # offline`)
		fmt.Fprintln(out, "")
		fs.PrintDefaults()
	}

	model := fs.String("model", "", "path to a local model.onnx (offline mode)")
	repo := fs.String("repo", ichkil.DefaultRepoID, "Hugging Face model repo (default: ichkil/ichkil)")
	revision := fs.String("revision", ichkil.DefaultRevision, "branch/tag/commit to fetch (default: main)")
	asJSON := fs.Bool("json", false, "emit a JSON object instead of plain text")
	version := fs.Bool("version", false, "print version and exit")
	if err := fs.Parse(os.Args[1:]); err != nil {
		return 2
	}
	if *version {
		fmt.Printf("ichkil %s\n", ichkil.Version)
		return 0
	}

	// Positional text (everything after the flags).
	text := strings.Join(fs.Args(), " ")
	if text == "" {
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error: reading stdin:", err)
			return 2
		}
		text = string(data)
	}
	// Any non-empty text is accepted: non-Arabic passes through unchanged.
	if strings.TrimSpace(text) == "" {
		fmt.Fprintln(os.Stderr, "error: no input text")
		fs.Usage()
		return 2
	}

	ctx := context.Background()
	var out string
	if *model != "" {
		d, err := ichkil.NewFromFiles(*model)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			return 1
		}
		defer d.Close()
		out, err = d.Diacritize(text)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			return 1
		}
	} else {
		opts := []ichkil.Option{
			ichkil.WithRepoID(*repo),
			ichkil.WithRevision(*revision),
		}
		d, err := ichkil.New(ctx, opts...)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			return 1
		}
		defer d.Close()
		out, err = d.Diacritize(text)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			return 1
		}
	}

	if *asJSON {
		doc, err := json.Marshal(struct {
			Input  string `json:"input"`
			Output string `json:"output"`
		}{Input: strings.TrimSpace(text), Output: out})
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			return 1
		}
		fmt.Println(string(doc))
	} else {
		fmt.Println(out)
	}
	return 0
}
