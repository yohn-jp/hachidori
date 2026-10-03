package e2e

import (
	"debug/pe"
	"fmt"
)

// checkPE reports whether path is a Windows amd64 PE image.
func checkPE(path string) error {
	f, err := pe.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if f.FileHeader.Machine != pe.IMAGE_FILE_MACHINE_AMD64 {
		return fmt.Errorf("machine 0x%x is not amd64", f.FileHeader.Machine)
	}
	return nil
}
