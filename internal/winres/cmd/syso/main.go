// Command syso writes the Windows icon resource object for an .ico file.
//
//	go run ./internal/winres/cmd/syso -ico assets/icons/hachidori.ico -o cmd/hachidori/rsrc_windows_amd64.syso
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/yohn-jp/hachidori/internal/winres"
)

func main() {
	ico := flag.String("ico", "", "source .ico file")
	out := flag.String("o", "", "output .syso file")
	flag.Parse()
	if *ico == "" || *out == "" {
		flag.Usage()
		os.Exit(2)
	}
	if err := run(*ico, *out); err != nil {
		fmt.Fprintln(os.Stderr, "syso:", err)
		os.Exit(1)
	}
}

func run(ico, out string) error {
	b, err := os.ReadFile(ico)
	if err != nil {
		return err
	}
	obj, err := winres.IconSyso(b, winres.MachineAMD64)
	if err != nil {
		return err
	}
	return os.WriteFile(out, obj, 0o644)
}
