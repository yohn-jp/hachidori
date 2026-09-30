package main

// rsrc_windows_amd64.syso embeds assets/icons/hachidori.ico as the Windows
// executable's application icon; see internal/winres.
//go:generate go run ../../internal/winres/cmd/syso -ico ../../assets/icons/hachidori.ico -o rsrc_windows_amd64.syso
