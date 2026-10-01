package setup

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/subprocess"
)

// uvTool is the verified Hachidori-managed uv executable.
type uvTool struct {
	h   home.Home
	exe string // absolute path under HACHIDORI_HOME/tools/uv/<version>/
	sha string // pinned executable digest
	log io.Writer
}

func exeName(name string) string {
	if runtime.GOOS == "windows" {
		return name + ".exe"
	}
	return name
}

// uvDir is the private uv tool directory for the pinned version.
func uvDir(h home.Home) string { return h.Path("tools", "uv", uvVersion) }

// ensureUV bootstraps the pinned uv under HACHIDORI_HOME: an existing
// executable is reused only if it matches the pinned digest; otherwise the
// pinned release archive is downloaded (or reused from packages/), verified,
// and the executable is extracted, verified and published atomically.
// A bootstrap failure is a setup failure: there is no fallback to another uv.
func ensureUV(h home.Home, log io.Writer, obs *Observer) (uvTool, error) {
	a, err := uvFor(platform())
	if err != nil {
		return uvTool{}, err
	}
	u := uvTool{h: h, exe: filepath.Join(uvDir(h), exeName("uv")), sha: a.BinarySHA256, log: log}
	if got, err := FileSHA256(u.exe); err == nil && got == a.BinarySHA256 {
		return u, nil
	} else if err == nil {
		fmt.Fprintf(log, "private uv %s digest mismatch (%s); replacing it from the pinned artifact\n", uvVersion, got)
	}
	archive := h.Path("packages", "uv-"+uvVersion+"-"+path.Base(a.URL))
	if err := fetch(a.URL, archive, a.SHA256, log, obs, Progress{Step: StepDownload, Detail: "uv " + uvVersion}); err != nil {
		return uvTool{}, fmt.Errorf("uv %s: %w", uvVersion, err)
	}
	if err := os.MkdirAll(uvDir(h), 0o755); err != nil {
		return uvTool{}, err
	}
	obs.step(StepMaterialize, "extracting uv "+uvVersion)
	tmp := u.exe + ".part"
	if err := extractMember(archive, a.Member, tmp); err != nil {
		os.Remove(tmp)
		return uvTool{}, fmt.Errorf("uv %s: %w", uvVersion, err)
	}
	if got, err := FileSHA256(tmp); err != nil || got != a.BinarySHA256 {
		os.Remove(tmp)
		return uvTool{}, fmt.Errorf("uv %s: extracted executable sha256 %s, want %s", uvVersion, got, a.BinarySHA256)
	}
	if err := os.Rename(tmp, u.exe); err != nil {
		return uvTool{}, err
	}
	fmt.Fprintf(log, "private uv %s ready: %s\n", uvVersion, u.exe)
	return u, nil
}

// env is the complete, explicitly constructed environment of uv. Nothing
// from the parent environment is inherited apart from what home.Env passes
// (OS essentials, proxy/CA settings), so user uv/pip configuration, indexes
// and interpreters cannot influence materialization.
func (u uvTool) env(extra ...string) []string {
	env := u.h.Env(filepath.Dir(u.exe), false)
	env = append(env,
		"UV_NO_CONFIG=1",
		"UV_CACHE_DIR="+u.h.Path("cache", "uv"),
		"UV_PYTHON_INSTALL_DIR="+filepath.Join(uvDir(u.h), "python"),
		"UV_PYTHON_BIN_DIR="+filepath.Join(uvDir(u.h), "bin"),
		"UV_TOOL_DIR="+filepath.Join(uvDir(u.h), "tool"),
		"UV_MANAGED_PYTHON=1",
		"UV_NO_PROGRESS=1",
	)
	return append(env, extra...)
}

// run verifies the pinned executable digest and invokes uv by absolute path.
func (u uvTool) run(dir string, env []string, args ...string) error {
	if got, err := FileSHA256(u.exe); err != nil || got != u.sha {
		return fmt.Errorf("private uv %s failed verification before execution (sha256 %s, want %s)", u.exe, got, u.sha)
	}
	fmt.Fprintf(u.log, "uv %s\n", strings.Join(args, " "))
	cmd := exec.Command(u.exe, args...)
	subprocess.Configure(cmd)
	cmd.Env, cmd.Dir = env, dir
	cmd.Stdout, cmd.Stderr = u.log, u.log
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("uv %s: %w", args[0], err)
	}
	return nil
}

// extractMember extracts one regular file from a verified .tar.gz or .zip archive.
func extractMember(archive, member, dst string) error {
	var r io.Reader
	if strings.HasSuffix(archive, ".zip") {
		z, err := zip.OpenReader(archive)
		if err != nil {
			return err
		}
		defer z.Close()
		for _, f := range z.File {
			if f.Name == member && f.Mode().IsRegular() {
				rc, err := f.Open()
				if err != nil {
					return err
				}
				defer rc.Close()
				r = rc
				break
			}
		}
	} else {
		f, err := os.Open(archive)
		if err != nil {
			return err
		}
		defer f.Close()
		gz, err := gzip.NewReader(f)
		if err != nil {
			return err
		}
		tr := tar.NewReader(gz)
		for r == nil {
			hdr, err := tr.Next()
			if err == io.EOF {
				break
			}
			if err != nil {
				return err
			}
			if hdr.Name == member && hdr.Typeflag == tar.TypeReg {
				r = tr
			}
		}
	}
	if r == nil {
		return fmt.Errorf("%s: %s not found", filepath.Base(archive), member)
	}
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
	if err != nil {
		return err
	}
	_, err = io.Copy(out, r)
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	return err
}
