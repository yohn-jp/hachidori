package setup

import (
	"archive/tar"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFetchVerifiesDigest(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "payload") }))
	defer srv.Close()
	sum := sha256.Sum256([]byte("payload"))
	dst := filepath.Join(t.TempDir(), "a", "f")
	if err := fetch(srv.URL, dst, hex.EncodeToString(sum[:]), io.Discard); err != nil {
		t.Fatal(err)
	}
	bad := filepath.Join(t.TempDir(), "g")
	if err := fetch(srv.URL, bad, strings.Repeat("0", 64), io.Discard); err == nil {
		t.Fatal("digest mismatch accepted")
	}
	if _, err := os.Stat(bad); !os.IsNotExist(err) {
		t.Fatal("unverified file left behind")
	}
}

func TestExtractRejectsTraversal(t *testing.T) {
	for _, hdr := range []*tar.Header{
		{Name: "../evil", Typeflag: tar.TypeReg, Mode: 0o644},
		{Name: "python/link", Typeflag: tar.TypeSymlink, Linkname: "../../etc/passwd"},
	} {
		p := filepath.Join(t.TempDir(), "a.tar.gz")
		f, _ := os.Create(p)
		gz := gzip.NewWriter(f)
		tw := tar.NewWriter(gz)
		tw.WriteHeader(hdr)
		tw.Close()
		gz.Close()
		f.Close()
		if err := extractTarGz(p, t.TempDir()); err == nil {
			t.Errorf("%s accepted", hdr.Name)
		}
	}
}

func TestPinsArePresent(t *testing.T) {
	for plat, d := range pythonDists {
		if len(d.SHA256) != 64 || !strings.HasPrefix(d.URL, "https://") {
			t.Errorf("%s: bad pin", plat)
		}
	}
	for rel, sum := range Model.Files {
		if len(sum) != 64 {
			t.Errorf("%s: bad digest", rel)
		}
	}
	for _, p := range packages {
		if !strings.Contains(p, "==") {
			t.Errorf("%s not exactly pinned", p)
		}
	}
	if _, err := RuntimeName("rocm"); err == nil {
		t.Error("unknown device accepted")
	}
}
