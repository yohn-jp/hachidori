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

func TestFileOpenDialogIIDMatchesWindowsSDK(t *testing.T) {
	want := [8]byte{0xBE, 0x02, 0x9D, 0x96, 0x95, 0x32, 0xD9, 0x60}
	if iidFileOpenDialog.Data1 != 0xD57C7288 ||
		iidFileOpenDialog.Data2 != 0xD4AD ||
		iidFileOpenDialog.Data3 != 0x4768 ||
		iidFileOpenDialog.Data4 != want {
		t.Fatalf("IID_IFileOpenDialog = %08X-%04X-%04X-%X, want D57C7288-D4AD-4768-BE02-9D969532D960",
			iidFileOpenDialog.Data1, iidFileOpenDialog.Data2, iidFileOpenDialog.Data3, iidFileOpenDialog.Data4)
	}
}
