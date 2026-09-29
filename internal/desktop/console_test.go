package desktop

import "testing"

func TestShouldHideOwnedConsole(t *testing.T) {
	const self uint32 = 42
	tests := []struct {
		name     string
		current  uint32
		attached []uint32
		want     bool
	}{
		{"owned console", self, []uint32{self}, true},
		{"shared terminal", self, []uint32{7, self}, false},
		{"other process only", self, []uint32{7}, false},
		{"no process id", 0, []uint32{self}, false},
		{"no attached process", self, nil, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := shouldHideOwnedConsole(tt.current, tt.attached); got != tt.want {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
		})
	}
}
