package winres

import (
	"bytes"
	"debug/pe"
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

var (
	icoPath  = filepath.Join("..", "..", "assets", "icons", "hachidori.ico")
	sysoPath = filepath.Join("..", "..", "cmd", "hachidori", "rsrc_windows_amd64.syso")
)

func readFile(t *testing.T, p string) []byte {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// TestCommittedSysoMatchesIcon fails when the committed resource object is
// not the generator output for the committed icon; rerun
// `go generate ./cmd/hachidori`.
func TestCommittedSysoMatchesIcon(t *testing.T) {
	want, err := IconSyso(readFile(t, icoPath), MachineAMD64)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(readFile(t, sysoPath), want) {
		t.Fatalf("%s is stale; run go generate ./cmd/hachidori", sysoPath)
	}
}

func TestParseICORejectsMalformed(t *testing.T) {
	ico := readFile(t, icoPath)
	for name, b := range map[string][]byte{
		"empty":     nil,
		"not icon":  {0, 0, 2, 0, 1, 0},
		"no images": {0, 0, 1, 0, 0, 0},
		"short dir": ico[:6+16],
		"truncated": ico[:len(ico)-1],
	} {
		if _, err := ParseICO(b); err == nil {
			t.Errorf("%s: ParseICO accepted malformed input", name)
		}
	}
}

// TestWindowsExecutableEmbedsIcon builds the Windows amd64 executable the way
// the development release does (plain go build of ./cmd/hachidori) and checks
// that its resource directory holds the icon group and every .ico image
// byte-for-byte.
func TestWindowsExecutableEmbedsIcon(t *testing.T) {
	if testing.Short() {
		t.Skip("cross-builds cmd/hachidori")
	}
	gobin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go tool not on PATH")
	}
	exe := filepath.Join(t.TempDir(), "hachidori.exe")
	cmd := exec.Command(gobin, "build", "-o", exe, "./cmd/hachidori")
	cmd.Dir = filepath.Join("..", "..")
	cmd.Env = append(os.Environ(), "GOOS=windows", "GOARCH=amd64", "CGO_ENABLED=0")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("go build: %v\n%s", err, out)
	}

	res, err := peResources(exe)
	if err != nil {
		t.Fatal(err)
	}
	imgs, err := ParseICO(readFile(t, icoPath))
	if err != nil {
		t.Fatal(err)
	}
	if got := res[key{rtGroupIcon, IconGroupID}]; !bytes.Equal(got, GroupIcon(imgs)) {
		t.Fatalf("RT_GROUP_ICON %d missing or different (%d bytes)", IconGroupID, len(got))
	}
	for i, im := range imgs {
		if got := res[key{rtIcon, uint32(i + 1)}]; !bytes.Equal(got, im.Data) {
			t.Errorf("RT_ICON %d missing or different from ico image %d", i+1, i)
		}
	}
	if len(res) != len(imgs)+1 {
		t.Errorf("executable has %d resources, want %d", len(res), len(imgs)+1)
	}
}

type key struct{ typ, id uint32 }

// peResources returns the data of every (type, id) resource of a PE file,
// taking the first language of each.
func peResources(path string) (map[key][]byte, error) {
	f, err := pe.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	oh, ok := f.OptionalHeader.(*pe.OptionalHeader64)
	if !ok || len(oh.DataDirectory) <= pe.IMAGE_DIRECTORY_ENTRY_RESOURCE {
		return nil, fmt.Errorf("not a PE32+ image")
	}
	rva := oh.DataDirectory[pe.IMAGE_DIRECTORY_ENTRY_RESOURCE].VirtualAddress
	if rva == 0 {
		return nil, fmt.Errorf("no resource directory")
	}
	var sec *pe.Section
	for _, s := range f.Sections {
		if rva >= s.VirtualAddress && rva < s.VirtualAddress+s.VirtualSize {
			sec = s
		}
	}
	if sec == nil {
		return nil, fmt.Errorf("resource directory outside any section")
	}
	data, err := sec.Data()
	if err != nil {
		return nil, err
	}
	base := rva - sec.VirtualAddress
	le := binary.LittleEndian
	u32 := func(off uint32) (uint32, error) {
		if uint64(off)+4 > uint64(len(data)) {
			return 0, fmt.Errorf("offset %#x out of range", off)
		}
		return le.Uint32(data[off:]), nil
	}
	entries := func(dir uint32) ([][2]uint32, error) {
		hdr, err := u32(base + dir + 12)
		if err != nil {
			return nil, err
		}
		n := hdr>>16 + hdr&0xffff
		var es [][2]uint32
		for i := uint32(0); i < n; i++ {
			id, err := u32(base + dir + 16 + 8*i)
			if err != nil {
				return nil, err
			}
			off, err := u32(base + dir + 16 + 8*i + 4)
			if err != nil {
				return nil, err
			}
			es = append(es, [2]uint32{id, off})
		}
		return es, nil
	}
	const subdir = 0x80000000
	out := map[key][]byte{}
	types, err := entries(0)
	if err != nil {
		return nil, err
	}
	for _, t := range types {
		names, err := entries(t[1] &^ subdir)
		if err != nil {
			return nil, err
		}
		for _, n := range names {
			langs, err := entries(n[1] &^ subdir)
			if err != nil || len(langs) == 0 {
				return nil, fmt.Errorf("resource %d/%d: no language entry", t[0], n[0])
			}
			entry := base + langs[0][1]
			drva, err := u32(entry)
			if err != nil {
				return nil, err
			}
			size, err := u32(entry + 4)
			if err != nil {
				return nil, err
			}
			start := drva - sec.VirtualAddress
			if uint64(start)+uint64(size) > uint64(len(data)) {
				return nil, fmt.Errorf("resource %d/%d data out of range", t[0], n[0])
			}
			out[key{t[0], n[0]}] = data[start : start+size]
		}
	}
	return out, nil
}
