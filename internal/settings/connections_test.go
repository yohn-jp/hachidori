package settings

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yohn-jp/hachidori/internal/setup"
	"github.com/yohn-jp/hachidori/internal/tunnel"
)

var nixos = Connection{Name: "nixos-dev", Destination: "dev@nixos", RemoteBind: "127.0.0.1", RemoteBindMode: ConnectionPinned, RemotePort: 7843, RemotePortMode: ConnectionPinned, LocalPort: 7843, LocalPortMode: ConnectionPinned}

func TestConnectionCreateEditRemovePersistAcrossRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state", "settings.json")
	s := &Store{Path: path}
	if err := s.SaveConnection(nixos); err != nil {
		t.Fatal(err)
	}
	other := Connection{Name: "a-box", Destination: "buildhost", RemoteBind: "::1", RemoteBindMode: ConnectionPinned, RemotePort: 9000, RemotePortMode: ConnectionPinned, LocalPort: 7843, LocalPortMode: ConnectionPinned}
	if err := s.SaveConnection(other); err != nil {
		t.Fatal(err)
	}
	edited := nixos
	edited.RemotePort = 7900
	if err := s.SaveConnection(edited); err != nil {
		t.Fatal(err)
	}
	got, err := (&Store{Path: path}).Connections() // a new process reads the file
	if err != nil || len(got) != 2 || withoutResolution(got[0]) != other || withoutResolution(got[1]) != edited {
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
	if got, _ := (&Store{Path: path}).Connections(); len(got) != 1 || withoutResolution(got[0]) != edited {
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
	if c, _ := s.Connections(); len(c) != 1 || withoutResolution(c[0]) != nixos {
		t.Fatalf("saving defaults lost the profile: %+v", c)
	}
}

func TestAutoResolutionKeepsIntentSeparateFromConcreteSpec(t *testing.T) {
	endpoint := tunnel.LocalEndpoint{Host: "127.0.0.1", Port: 9123}
	auto := Connection{
		Name: "auto-dev", Destination: "devhost",
		RemoteBindMode: ConnectionAuto, RemotePortMode: ConnectionAuto, LocalPortMode: ConnectionAuto,
	}
	resolved, err := auto.Resolve(endpoint)
	if err != nil {
		t.Fatal(err)
	}
	want := tunnel.Spec{Destination: "devhost", RemoteBind: "127.0.0.1", RemotePort: 9123, LocalPort: 9123}
	if resolved.Spec != want {
		t.Fatalf("resolved spec %+v, want %+v", resolved.Spec, want)
	}
	if resolved.Intent.RemoteBindMode != ConnectionAuto || resolved.Intent.RemotePortMode != ConnectionAuto || resolved.Intent.LocalPortMode != ConnectionAuto || resolved.Intent.RemoteBind != "" || resolved.Intent.RemotePort != 0 || resolved.Intent.LocalPort != 0 {
		t.Fatalf("resolution overwrote stored intent: %+v", resolved.Intent)
	}

	// The production Settings store attaches the current managed endpoint to
	// returned profiles, so the existing Connection.Spec dashboard seam sees
	// the same concrete, validated result.
	s := &Store{Path: filepath.Join(t.TempDir(), "settings.json")}
	if err := s.SetLocalEndpoint(endpoint); err != nil {
		t.Fatal(err)
	}
	if err := s.SaveConnection(auto); err != nil {
		t.Fatal(err)
	}
	stored, err := os.ReadFile(s.Path)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"remote_bind_mode": "auto"`, `"remote_port_mode": "auto"`, `"local_port_mode": "auto"`} {
		if !strings.Contains(string(stored), want) {
			t.Errorf("stored Auto intent lacks %s:\n%s", want, stored)
		}
	}
	profiles, err := s.Connections()
	if err != nil || len(profiles) != 1 || profiles[0].Spec() != want || profiles[0].Endpoint() != "http://127.0.0.1:9123" {
		t.Fatalf("managed connection spec: %+v, %v", profiles, err)
	}
}

func TestAutoResolutionAllowsPinnedOverridesAndRejectsUnsafeValues(t *testing.T) {
	endpoint := tunnel.LocalEndpoint{Host: "127.0.0.1", Port: 9123}
	profile := Connection{
		Name: "override-dev", Destination: "devhost",
		RemoteBind: "::1", RemoteBindMode: ConnectionPinned,
		RemotePort: 9001, RemotePortMode: ConnectionPinned,
		LocalPortMode: ConnectionAuto,
	}
	resolved, err := profile.Resolve(endpoint)
	if err != nil {
		t.Fatal(err)
	}
	if want := (tunnel.Spec{Destination: "devhost", RemoteBind: "::1", RemotePort: 9001, LocalPort: 9123}); resolved.Spec != want {
		t.Fatalf("pinned resolution %+v, want %+v", resolved.Spec, want)
	}

	for name, invalid := range map[string]Connection{
		"public pinned bind": {
			Name: "unsafe", Destination: "devhost", RemoteBind: "0.0.0.0", RemoteBindMode: ConnectionPinned,
			RemotePort: 9123, RemotePortMode: ConnectionAuto, LocalPortMode: ConnectionAuto,
		},
		"invalid pinned port": {
			Name: "bad-port", Destination: "devhost", RemoteBindMode: ConnectionAuto,
			RemotePort: 0, RemotePortMode: ConnectionPinned, LocalPortMode: ConnectionAuto,
		},
		"auto with stale value": {
			Name: "ambiguous", Destination: "devhost", RemoteBind: "127.0.0.1", RemoteBindMode: ConnectionAuto,
			RemotePortMode: ConnectionAuto, LocalPortMode: ConnectionAuto,
		},
	} {
		if _, err := invalid.Resolve(endpoint); err == nil {
			t.Errorf("%s was silently resolved", name)
		}
	}
	if _, err := profile.Resolve(tunnel.LocalEndpoint{Host: "0.0.0.0", Port: 9123}); err == nil {
		t.Fatal("unsafe managed endpoint was accepted")
	}
}

func TestLegacyConnectionValuesLoadAsPinned(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	body := `{"schema":"hachidori.settings/1","runtime_defaults":{},"connections":[{"name":"legacy","destination":"devhost","remote_bind":"::1","remote_port":9000,"local_port":7843}]}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	profiles, err := (&Store{Path: path}).Connections()
	if err != nil || len(profiles) != 1 {
		t.Fatalf("legacy profile: %+v %v", profiles, err)
	}
	c := profiles[0]
	if c.RemoteBindMode != ConnectionPinned || c.RemotePortMode != ConnectionPinned || c.LocalPortMode != ConnectionPinned {
		t.Fatalf("legacy modes: %+v", c)
	}
	if want := (tunnel.Spec{Destination: "devhost", RemoteBind: "::1", RemotePort: 9000, LocalPort: 7843}); c.Spec() != want {
		t.Fatalf("legacy spec %+v, want %+v", c.Spec(), want)
	}
	if b, _ := os.ReadFile(path); string(b) != body {
		t.Fatal("reading a legacy profile rewrote settings.json")
	}
}

func withoutResolution(c Connection) Connection {
	c.localEndpoint = tunnel.LocalEndpoint{}
	return c
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
