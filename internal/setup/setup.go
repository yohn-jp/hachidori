package setup

import (
	"archive/tar"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/worker/py"
)

// Run materializes the pinned runtime and model and activates them.
// Existing materialized versions are immutable and reused as-is.
func Run(h home.Home, device string, log io.Writer) error {
	name, err := RuntimeName(device)
	if err != nil {
		return err
	}
	if err := h.Ensure(); err != nil {
		return err
	}
	if err := materializeRuntime(h, name, device, log); err != nil {
		return fmt.Errorf("runtime: %w", err)
	}
	if err := materializeModel(h, log); err != nil {
		return fmt.Errorf("model: %w", err)
	}
	a := home.Active{Runtime: name, Model: ModelDirName(), Device: device}
	if err := home.WriteJSON(h.Path("state", "active-runtime.json"), a); err != nil {
		return err
	}
	fmt.Fprintf(log, "active: runtime=%s model=%s device=%s\n", a.Runtime, a.Model, a.Device)
	return nil
}

func materializeRuntime(h home.Home, name, device string, log io.Writer) error {
	final := h.Path("runtime", name)
	if _, err := os.Stat(filepath.Join(final, "manifest.json")); err == nil {
		fmt.Fprintf(log, "runtime %s already materialized\n", name)
		return nil
	}
	dist, ok := pythonDists[platform()]
	if !ok {
		return fmt.Errorf("no pinned Python distribution for %s", platform())
	}
	flavor := flavors[device]
	stage := h.Path("runtime", ".staging-"+name)
	if err := os.RemoveAll(stage); err != nil {
		return err
	}
	if err := os.MkdirAll(stage, 0o755); err != nil {
		return err
	}

	archive := h.Path("packages", filepath.Base(strings.ReplaceAll(dist.URL, "%2B", "+")))
	if err := fetch(dist.URL, archive, dist.SHA256, log); err != nil {
		return err
	}
	fmt.Fprintf(log, "extracting %s\n", filepath.Base(archive))
	if err := extractTarGz(archive, stage); err != nil {
		return err
	}
	python := filepath.Join(stage, filepath.FromSlash(dist.RelPath))

	reqs := filepath.Join(stage, "requirements.txt")
	lines := append([]string{flavor.Torch}, packages...)
	if err := os.WriteFile(reqs, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		return err
	}
	env := h.Env(filepath.Dir(python), false)
	pip := func(args ...string) *exec.Cmd {
		cmd := exec.Command(python, append([]string{"-I", "-m", "pip"}, args...)...)
		cmd.Env, cmd.Dir = env, stage
		return cmd
	}
	fmt.Fprintf(log, "installing pinned packages (%s)\n", flavor.Torch)
	cmd := pip("install", "--no-input", "--only-binary=:all:", "--no-warn-script-location",
		"--index-url", "https://pypi.org/simple", "--extra-index-url", flavor.Index, "-r", reqs)
	cmd.Stdout, cmd.Stderr = log, log
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("pip install: %w", err)
	}
	freeze, err := pip("freeze", "--all").Output()
	if err != nil {
		return fmt.Errorf("pip freeze: %w", err)
	}

	if err := os.MkdirAll(filepath.Join(stage, "worker"), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(stage, "worker", "hachidori_worker.py"), py.Script, 0o644); err != nil {
		return err
	}
	sum := sha256.Sum256(py.Script)
	m := home.RuntimeManifest{
		Version: RuntimeVersion, Flavor: flavor.Name, Platform: platform(), PythonVersion: pythonVersion,
		PythonArchive: dist.Artifact, PythonRelPath: dist.RelPath,
		Packages: lines, PackageIndexes: []string{"https://pypi.org/simple", flavor.Index},
		Installed: strings.Fields(string(freeze)),
		Worker:    map[string]string{"worker/hachidori_worker.py": hex.EncodeToString(sum[:])},
	}
	if err := home.WriteJSON(filepath.Join(stage, "manifest.json"), m); err != nil {
		return err
	}
	return os.Rename(stage, final)
}

func materializeModel(h home.Home, log io.Writer) error {
	final := h.Path("models", filepath.FromSlash(ModelDirName()))
	if _, err := os.Stat(filepath.Join(final, "hachidori-model.json")); err == nil {
		fmt.Fprintf(log, "model %s@%s already materialized\n", Model.Repo, Model.Revision[:12])
		return nil
	}
	stage := final + ".staging"
	if err := os.RemoveAll(stage); err != nil {
		return err
	}
	for rel, want := range Model.Files {
		url := "https://huggingface.co/" + Model.Repo + "/resolve/" + Model.Revision + "/" + rel
		if err := fetch(url, filepath.Join(stage, filepath.FromSlash(rel)), want, log); err != nil {
			return err
		}
	}
	if err := home.WriteJSON(filepath.Join(stage, "hachidori-model.json"), Model); err != nil {
		return err
	}
	return os.Rename(stage, final)
}

// fetch downloads url to dst and verifies its SHA-256. A present file with
// the right digest is reused.
func fetch(url, dst, want string, log io.Writer) error {
	if got, err := FileSHA256(dst); err == nil && got == want {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	fmt.Fprintf(log, "downloading %s\n", url)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	tmp := dst + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	hash := sha256.New()
	_, err = io.Copy(io.MultiWriter(f, hash), resp.Body)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(tmp)
		return err
	}
	if got := hex.EncodeToString(hash.Sum(nil)); got != want {
		os.Remove(tmp)
		return fmt.Errorf("%s: sha256 mismatch: got %s want %s", url, got, want)
	}
	return os.Rename(tmp, dst)
}

// FileSHA256 hashes a file.
func FileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// extractTarGz extracts a verified archive, refusing entries that escape dst.
func extractTarGz(archive, dst string) error {
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
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		target := filepath.Join(dst, filepath.FromSlash(hdr.Name))
		if !within(dst, target) {
			return fmt.Errorf("archive entry escapes destination: %s", hdr.Name)
		}
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			out, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, os.FileMode(hdr.Mode)&0o755|0o644)
			if err != nil {
				return err
			}
			_, err = io.Copy(out, tr)
			if cerr := out.Close(); err == nil {
				err = cerr
			}
			if err != nil {
				return err
			}
		case tar.TypeSymlink:
			if filepath.IsAbs(hdr.Linkname) || !within(dst, filepath.Join(filepath.Dir(target), hdr.Linkname)) {
				return fmt.Errorf("archive symlink escapes destination: %s", hdr.Name)
			}
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			if err := os.Symlink(hdr.Linkname, target); err != nil {
				return err
			}
		}
	}
}

func within(root, p string) bool {
	rel, err := filepath.Rel(root, p)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}
