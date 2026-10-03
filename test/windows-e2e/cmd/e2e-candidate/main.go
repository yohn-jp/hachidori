// Command e2e-candidate writes and verifies the Windows E2E candidate
// manifest.
//
//	e2e-candidate write  -dir DIR -file hachidori.exe -commit SHA
//	e2e-candidate verify -dir DIR [-commit SHA] [-sha256 HEX]
//
// Both print `sha256=<hex>` and `source_commit=<sha>` lines so a workflow can
// append them to $GITHUB_OUTPUT. verify exits non-zero on any identity
// mismatch.
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"

	"github.com/yohn-jp/hachidori/test/windows-e2e/e2e"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: e2e-candidate write|verify [flags]")
		os.Exit(2)
	}
	switch os.Args[1] {
	case "write":
		os.Exit(write(os.Args[2:]))
	case "verify":
		os.Exit(verify(os.Args[2:]))
	default:
		fmt.Fprintf(os.Stderr, "e2e-candidate: unknown command %q\n", os.Args[1])
		os.Exit(2)
	}
}

func write(args []string) int {
	fs := flag.NewFlagSet("write", flag.ContinueOnError)
	dir := fs.String("dir", "", "candidate directory holding the executable")
	file := fs.String("file", "hachidori.exe", "executable file name inside dir")
	commit := fs.String("commit", "", "source commit SHA being certified")
	goos := fs.String("goos", runtime.GOOS, "GOOS the executable was built for")
	goarch := fs.String("goarch", runtime.GOARCH, "GOARCH the executable was built for")
	goVersion := fs.String("go-version", runtime.Version(), "Go toolchain version")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *dir == "" {
		fmt.Fprintln(os.Stderr, "e2e-candidate write: -dir is required")
		return 2
	}
	m, err := e2e.WriteCandidate(*dir, *file, *commit, *goos, *goarch, *goVersion)
	if err != nil {
		fmt.Fprintln(os.Stderr, "e2e-candidate write:", err)
		return 1
	}
	printIdentity(m)
	return 0
}

func verify(args []string) int {
	fs := flag.NewFlagSet("verify", flag.ContinueOnError)
	dir := fs.String("dir", "", "candidate directory")
	commit := fs.String("commit", "", "expected source commit SHA")
	sum := fs.String("sha256", "", "expected executable SHA-256")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *dir == "" {
		fmt.Fprintln(os.Stderr, "e2e-candidate verify: -dir is required")
		return 2
	}
	c, err := e2e.LoadCandidate(*dir, e2e.Expect{SourceCommit: *commit, SHA256: *sum})
	if err != nil {
		fmt.Fprintln(os.Stderr, "e2e-candidate verify:", err)
		return 1
	}
	printIdentity(c.Manifest)
	return 0
}

func printIdentity(m e2e.Manifest) {
	fmt.Printf("sha256=%s\n", m.SHA256)
	fmt.Printf("source_commit=%s\n", m.SourceCommit)
	fmt.Printf("file=%s\n", m.File)
}
