//go:build linux

package setup

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

func hostMemory() Memory {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return Memory{}
	}
	defer f.Close()
	var m Memory
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		name, rest, ok := strings.Cut(sc.Text(), ":")
		if !ok {
			continue
		}
		fields := strings.Fields(rest)
		if len(fields) != 2 || fields[1] != "kB" {
			continue
		}
		kb, err := strconv.ParseUint(fields[0], 10, 64)
		if err != nil {
			continue
		}
		switch name {
		case "MemTotal":
			m.Total, m.TotalKnown = kb<<10, true
		case "MemAvailable":
			m.Available, m.AvailableKnown = kb<<10, true
		}
	}
	return m
}
