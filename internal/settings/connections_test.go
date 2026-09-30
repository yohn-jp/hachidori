package settings

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/setup"
)

var nixos = Connection{Name: "nixos-dev", Destination: "dev@nixos", RemoteBind: "127.0.0.1", RemotePort: 7843, LocalPort: 7843}

func TestConnectionCreateEditRemovePersistAcrossRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state", "settings.json")
	s := &Store{Path: path}
	if err := s.SaveConnection(nixos); err != nil {
		t.Fatal(err)
	}
	other := Connection{Name: "a-box", Destination: "buildhost", RemoteBind: "::1", RemotePort: 9000, LocalPort: 7843}
	if err := s.SaveConnection(other); err != nil {
		t.Fatal(err)
	}
	edited := nixos
	edited.RemotePort = 7900
	if err := s.SaveConnection(edited); err != nil {
		t.Fatal(err)
	}
	got, err := (&Store{Path: path}).Connections() // a new process reads the file
	if err != nil || len(got) != 2 || got[0] != other || got[1] != edited {
		t.Fatalf("after restart: %+v %v", got, err)
	}
	if got[1].Endpoint() != "http://127.0.0.1:7900" || got[0].Endpoint() != "http://[::1]:9000" {
		t.Fatalf("endpoints %q %q", got[1].Endpoint(), got[0].Endpoint())
	}
	if err := s.RemoveConnection("a-box"); err != nil {
		t.Fatal(err)
	}
	if err := s.RemoveConnection("a-box"); err == nil {
		t.Fatal("removing a missing profile succeeded")
	}
	if got, _ := (&Store{Path: path}).Connections(); len(got) != 1 || got[0] != edited {
		t.Fatalf("after remove: %+v", got)
	}
}

func TestConnectionValidationRejectsWithoutWriting(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	s := &Store{Path: path}
	base := Connection{Name: "n", Destination: "dev@nixos", RemoteBind: "127.0.0.1", RemotePort: 1, LocalPort: 1}
	mut := func(f func(*Connection)) Connection { c := base; f(&c); return c }
	for name, c := range map[string]Connection{
		"empty name":  mut(func(c *Connection) { c.Name = "" }),
		"path name":   mut(func(c *Connection) { c.Name = "../x" }),
		"option dest": mut(func(c *Connection) { c.Destination = "-oProxyCommand=calc" }),
		"public bind": mut(func(c *Connection) { c.RemoteBind = "0.0.0.0" }),
		"lan bind":    mut(func(c *Connection) { c.RemoteBind = "192.168.1.5" }),
		"remote port": mut(func(c *Connection) { c.RemotePort = 0 }),
		"local port":  mut(func(c *Connection) { c.LocalPort = 70000 }),
	} {
		if err := s.SaveConnection(c); err == nil {
			t.Errorf("%s: accepted %+v", name, c)
		}
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("invalid input wrote a file: %v", err)
	}
}

func TestConnectionRecordIsNonSecretAndDeterministic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	s := &Store{Path: path}
	if err := s.SaveConnection(nixos); err != nil {
		t.Fatal(err)
	}
	first, _ := os.ReadFile(path)
	if err := s.SaveConnection(nixos); err != nil {
		t.Fatal(err)
	}
	second, _ := os.ReadFile(path)
	if string(first) != string(second) {
		t.Fatal("same profile wrote different bytes")
	}
	for _, want := range []string{`"schema": "hachidori.settings/1"`, `"connections"`, `"name": "nixos-dev"`, `"destination": "dev@nixos"`} {
		if !strings.Contains(string(first), want) {
			t.Errorf("record lacks %s:\n%s", want, first)
		}
	}
	for _, bad := range []string{"key", "pass", "identity", "secret", "token", "known_hosts", "BEGIN"} {
		if strings.Contains(strings.ToLower(string(first)), strings.ToLower(bad)) {
			t.Errorf("record mentions %q:\n%s", bad, first)
		}
	}
}

// A settings.json written before profiles existed is the same schema with no
// profiles; profiles and runtime defaults never overwrite each other.
func TestProfilesMigrateFromPreProfileSettingsAndCoexistWithDefaults(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	old := "{\n  \"schema\": \"hachidori.settings/1\",\n  \"runtime_defaults\": {\n    \"device\": \"cpu\"\n  }\n}\n"
	if err := os.WriteFile(path, []byte(old), 0o644); err != nil {
		t.Fatal(err)
	}
	s := &Store{Path: path}
	if c, err := s.Connections(); err != nil || len(c) != 0 {
		t.Fatalf("old file: %+v %v", c, err)
	}
	if err := s.SaveConnection(nixos); err != nil {
		t.Fatal(err)
	}
	if d, _ := s.Defaults(); d.Device != DeviceCPU {
		t.Fatalf("saving a profile lost the defaults: %+v", d)
	}
	if err := s.SetDefaults(Defaults{Device: DeviceCUDA, Model: setup.DefaultModel}); err != nil {
		t.Fatal(err)
	}
	if c, _ := s.Connections(); len(c) != 1 || c[0] != nixos {
		t.Fatalf("saving defaults lost the profile: %+v", c)
	}
}

func TestInvalidStoredProfilesAreReportedAndLeftUntouched(t *testing.T) {
	dir := t.TempDir()
	one := `{"name":"n","destination":"h","remote_bind":"127.0.0.1","remote_port":1,"local_port":1}`
	for name, conn := range map[string]string{
		"public.json": `{"name":"n","destination":"h","remote_bind":"0.0.0.0","remote_port":1,"local_port":1}`,
		"dup.json":    one + "," + one,
	} {
		p := filepath.Join(dir, name)
		body := `{"schema":"hachidori.settings/1","runtime_defaults":{},"connections":[` + conn + `]}`
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		s := &Store{Path: p}
		if c, err := s.Connections(); err == nil || len(c) != 0 {
			t.Errorf("%s: %+v %v", name, c, err)
		}
		if err := s.SaveConnection(nixos); err == nil {
			t.Errorf("%s: save over an invalid record succeeded", name)
		}
		if b, _ := os.ReadFile(p); string(b) != body {
			t.Errorf("%s: file was modified", name)
		}
	}
}

func TestConnectionLimitAndMemoryStore(t *testing.T) {
	s := &Store{}
	for i := 0; i < MaxConnections; i++ {
		c := nixos
		c.Name = "c" + string(rune('a'+i/26)) + string(rune('a'+i%26))
		if err := s.SaveConnection(c); err != nil {
			t.Fatal(err)
		}
	}
	extra := nixos
	extra.Name = "one-too-many"
	if err := s.SaveConnection(extra); err == nil {
		t.Fatal("exceeded the profile limit")
	}
	if c, _ := s.Connections(); len(c) != MaxConnections {
		t.Fatalf("%d", len(c))
	}
}
