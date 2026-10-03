package desktopkit

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/yohn-jp/hachidori/internal/home"
)

// WriteLocator writes a valid bootstrap.json naming homeRoot at path, in the
// production record shape (home.Bootstrap). The home is not created, so a
// locator for a missing folder can be written too.
func WriteLocator(path, homeRoot string) ([]byte, error) {
	data, err := json.MarshalIndent(home.Bootstrap{Schema: home.BootstrapSchema, Home: homeRoot}, "", "  ")
	if err != nil {
		return nil, err
	}
	data = append(data, '\n')
	return data, WriteRaw(path, data)
}

// WriteRaw writes arbitrary bytes as the locator, creating its directory.
func WriteRaw(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o600)
}

// Unchanged verifies that the file at path still holds exactly want.
func Unchanged(path string, want []byte) error {
	got, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read %s: %w", filepath.Base(path), err)
	}
	if !bytes.Equal(got, want) {
		return fmt.Errorf("%s changed: %d bytes, want the original %d bytes", filepath.Base(path), len(got), len(want))
	}
	return nil
}

// CheckLocatorShape verifies the bootstrap contract: the record holds exactly
// the keys "schema" and "home", the schema is the current one, and home is
// wantHome.
func CheckLocatorShape(data []byte, wantHome string) error {
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return fmt.Errorf("bootstrap.json is not a JSON object: %w", err)
	}
	if len(m) != 2 {
		return fmt.Errorf("bootstrap.json has %d keys, want exactly schema and home", len(m))
	}
	if s, _ := m["schema"].(string); s != home.BootstrapSchema {
		return fmt.Errorf("bootstrap.json schema %q, want %q", s, home.BootstrapSchema)
	}
	h, ok := m["home"].(string)
	if !ok {
		return fmt.Errorf("bootstrap.json has no string home")
	}
	if h != wantHome {
		return fmt.Errorf("bootstrap.json home %q, want %q", h, wantHome)
	}
	return nil
}

// Corruption is one way the bootstrap locator can be unusable.
type Corruption struct {
	Name string
	Data []byte
}

// Corruptions are the corrupt locators the recovery scenario feeds the
// candidate. Each is rejected by the production locator parser; none is a first
// run.
func Corruptions(homeRoot string) []Corruption {
	quoted, _ := json.Marshal(homeRoot)
	return []Corruption{
		{Name: "empty", Data: nil},
		{Name: "binary-garbage", Data: []byte{0xff, 0xfe, 0x00, 0x01, 0x02, 0xfd}},
		{Name: "truncated-json", Data: []byte(`{"schema": "hachidori.bootstrap/1", "home": "C:\Use`)},
		{Name: "unknown-schema", Data: []byte(`{"schema": "hachidori.bootstrap/99", "home": ` + string(quoted) + "}\n")},
		{Name: "relative-home", Data: []byte(`{"schema": "hachidori.bootstrap/1", "home": "relative\\folder"}` + "\n")},
	}
}
