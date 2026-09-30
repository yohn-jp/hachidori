// Package winres builds the Windows resource object (.syso) that gives the
// hachidori executable its application icon.
//
// The Go toolchain links every rsrc_windows_<arch>.syso in a main package
// into the Windows executable, so a plain `go build ./cmd/hachidori` (and the
// development release workflow, which runs exactly that) embeds the icon. The
// committed object is generated from assets/icons/hachidori.ico by
// `go generate ./cmd/hachidori`; TestCommittedSysoMatchesIcon keeps the two in
// sync.
//
// The object is a COFF file with a single .rsrc section holding one
// RT_GROUP_ICON (IconGroupID) and one RT_ICON per image of the .ico. The
// .ico images and directory fields are copied verbatim; nothing is re-encoded.
package winres

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
)

// IconGroupID is the RT_GROUP_ICON resource id of the application icon. As
// the only (and so lowest) icon group it is also the icon Explorer shows for
// the executable.
const IconGroupID = 1

// MachineAMD64 is IMAGE_FILE_MACHINE_AMD64.
const MachineAMD64 = 0x8664

const (
	rtIcon      = 3
	rtGroupIcon = 14
	langEnUS    = 0x0409

	imageRelAMD64Addr32NB = 0x3
	imageSymClassStatic   = 3
	sectionCharacteristic = 0x40000040 // IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ
)

// Image is one image of an .ico file.
type Image struct {
	Width, Height, ColorCount, Reserved uint8
	Planes, BitCount                    uint16
	Data                                []byte
}

// ParseICO returns the images of an .ico file in directory order.
func ParseICO(ico []byte) ([]Image, error) {
	if len(ico) < 6 {
		return nil, errors.New("ico: short header")
	}
	le := binary.LittleEndian
	if le.Uint16(ico[0:]) != 0 || le.Uint16(ico[2:]) != 1 {
		return nil, errors.New("ico: not an icon file")
	}
	n := int(le.Uint16(ico[4:]))
	if n == 0 {
		return nil, errors.New("ico: no images")
	}
	if len(ico) < 6+16*n {
		return nil, errors.New("ico: short directory")
	}
	imgs := make([]Image, n)
	for i := range imgs {
		e := ico[6+16*i:]
		size, off := le.Uint32(e[8:]), le.Uint32(e[12:])
		if uint64(off)+uint64(size) > uint64(len(ico)) {
			return nil, fmt.Errorf("ico: image %d out of range", i)
		}
		imgs[i] = Image{Width: e[0], Height: e[1], ColorCount: e[2], Reserved: e[3],
			Planes: le.Uint16(e[4:]), BitCount: le.Uint16(e[6:]), Data: ico[off : off+size]}
	}
	return imgs, nil
}

// GroupIcon is the RT_GROUP_ICON (GRPICONDIR) data for imgs, whose RT_ICON
// ids are 1..len(imgs).
func GroupIcon(imgs []Image) []byte {
	var b bytes.Buffer
	w := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	w([3]uint16{0, 1, uint16(len(imgs))})
	for i, im := range imgs {
		w([4]uint8{im.Width, im.Height, im.ColorCount, im.Reserved})
		w([2]uint16{im.Planes, im.BitCount})
		w(uint32(len(im.Data)))
		w(uint16(i + 1))
	}
	return b.Bytes()
}

// IconSyso returns a COFF object for machine whose .rsrc section holds the
// icon of the .ico file as RT_GROUP_ICON IconGroupID.
func IconSyso(ico []byte, machine uint16) ([]byte, error) {
	imgs, err := ParseICO(ico)
	if err != nil {
		return nil, err
	}
	if machine != MachineAMD64 {
		return nil, fmt.Errorf("winres: unsupported machine %#x", machine)
	}

	// Resource leaves in directory order: RT_ICON 1..n, then RT_GROUP_ICON.
	type leaf struct {
		id   uint16
		data []byte
	}
	icons := make([]leaf, len(imgs))
	for i, im := range imgs {
		icons[i] = leaf{uint16(i + 1), im.Data}
	}
	types := []struct {
		id     uint16
		leaves []leaf
	}{
		{rtIcon, icons},
		{rtGroupIcon, []leaf{{IconGroupID, GroupIcon(imgs)}}},
	}

	// Layout: root dir, type dirs, name dirs, language dirs, data entries,
	// data. Directory offsets are section-relative; data entry offsets are
	// RVAs and so carry a relocation against the section symbol.
	const dirHeader, dirEntry, dataEntry = 16, 8, 16
	nLeaves := 0
	for _, t := range types {
		nLeaves += len(t.leaves)
	}
	rootSize := dirHeader + dirEntry*len(types)
	nameDirs := rootSize
	langDirs := nameDirs
	for _, t := range types {
		langDirs += dirHeader + dirEntry*len(t.leaves)
	}
	entries := langDirs + nLeaves*(dirHeader+dirEntry)
	dataStart := entries + nLeaves*dataEntry

	var sec bytes.Buffer
	w := func(v any) { _ = binary.Write(&sec, binary.LittleEndian, v) }
	dir := func(n int) { w([4]uint32{0, 0, 0, uint32(n) << 16}) } // no named entries, n id entries
	const subdir = 0x80000000

	dir(len(types))
	off := nameDirs
	for _, t := range types {
		w([2]uint32{uint32(t.id), uint32(off) | subdir})
		off += dirHeader + dirEntry*len(t.leaves)
	}
	leafIdx := 0
	for _, t := range types {
		dir(len(t.leaves))
		for _, l := range t.leaves {
			w([2]uint32{uint32(l.id), uint32(langDirs+leafIdx*(dirHeader+dirEntry)) | subdir})
			leafIdx++
		}
	}
	for i := 0; i < nLeaves; i++ {
		dir(1)
		w([2]uint32{langEnUS, uint32(entries + i*dataEntry)})
	}
	var relocs []uint32
	dataOff := dataStart
	for _, t := range types {
		for _, l := range t.leaves {
			relocs = append(relocs, uint32(sec.Len()))
			w([4]uint32{uint32(dataOff), uint32(len(l.data)), 0, 0})
			dataOff = align8(dataOff + len(l.data))
		}
	}
	for _, t := range types {
		for _, l := range t.leaves {
			sec.Write(l.data)
			sec.Write(make([]byte, align8(sec.Len())-sec.Len()))
		}
	}

	// COFF: file header, one section header, section data, relocations,
	// symbol table (the section symbol), empty string table.
	const fileHeader, sectionHeader, relocSize = 20, 40, 10
	rawPtr := fileHeader + sectionHeader
	relocPtr := rawPtr + sec.Len()
	symPtr := relocPtr + relocSize*len(relocs)

	var obj bytes.Buffer
	o := func(v any) { _ = binary.Write(&obj, binary.LittleEndian, v) }
	o(machine)
	o(uint16(1))      // NumberOfSections
	o(uint32(0))      // TimeDateStamp: reproducible
	o(uint32(symPtr)) // PointerToSymbolTable
	o(uint32(1))      // NumberOfSymbols
	o(uint16(0))      // SizeOfOptionalHeader
	o(uint16(0))      // Characteristics
	o([8]byte{'.', 'r', 's', 'r', 'c'})
	o(uint32(0))           // VirtualSize
	o(uint32(0))           // VirtualAddress
	o(uint32(sec.Len()))   // SizeOfRawData
	o(uint32(rawPtr))      // PointerToRawData
	o(uint32(relocPtr))    // PointerToRelocations
	o(uint32(0))           // PointerToLinenumbers
	o(uint16(len(relocs))) // NumberOfRelocations
	o(uint16(0))           // NumberOfLinenumbers
	o(uint32(sectionCharacteristic))
	obj.Write(sec.Bytes())
	for _, r := range relocs {
		o(r)         // VirtualAddress
		o(uint32(0)) // SymbolTableIndex: .rsrc
		o(uint16(imageRelAMD64Addr32NB))
	}
	o([8]byte{'.', 'r', 's', 'r', 'c'})
	o(uint32(0)) // Value
	o(int16(1))  // SectionNumber
	o(uint16(0)) // Type
	o(uint8(imageSymClassStatic))
	o(uint8(0))  // NumberOfAuxSymbols
	o(uint32(4)) // string table: size only
	return obj.Bytes(), nil
}

func align8(n int) int { return (n + 7) &^ 7 }
