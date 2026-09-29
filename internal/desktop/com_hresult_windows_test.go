//go:build windows

package desktop

import "testing"

func TestNormalizeHRESULTUsesCOM32BitDomain(t *testing.T) {
	const noInterface hresult = 0x80004002
	if got := normalizeHRESULT(uintptr(0xffffffff80004002)); got != noInterface {
		t.Fatalf("normalizeHRESULT = %#x, want %#x", got, noInterface)
	}
	if got := normalizeHRESULT(uintptr(0x800704c7)); got != hrCancelled {
		t.Fatalf("cancel HRESULT = %#x, want %#x", got, hrCancelled)
	}
	if got := normalizeHRESULT(0); got != sOK {
		t.Fatalf("success HRESULT = %#x, want %#x", got, sOK)
	}
}
