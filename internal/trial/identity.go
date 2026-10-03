// Package trial is the RAM-resident tuning-trial substrate.
//
// A Trial is an ephemeral, reversible composition of the model that is being
// measured: the resolved tuning plan (internal/home TuningPlan, the canonical
// policy model of layer-wise tuning) is applied to a resident worker's GPU
// model as a bounded delta, using transformed components cached in system RAM,
// without ever writing a model artifact. A Candidate is the reproducible
// record of a measured plan: its exact source, profile, plan and component
// set, with immutable measurement Evidence. A Variant stays the immutable
// Forge artifact; a Candidate becomes one only through Forge, from the exact
// plan it recorded.
//
// System RAM is a working and cache tier. Inference always runs on the
// accelerator against the assembled trial; nothing here offloads execution to
// the CPU.
package trial

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/yohn-jp/hachidori/internal/home"
	"github.com/yohn-jp/hachidori/internal/optimize"
)

const (
	// ComponentSchema versions the component identity derivation. A component
	// cached under another schema is never reused.
	ComponentSchema = "hachidori.trial-component/1"

	// TransformImplementation names the transformation code the worker runs to
	// build a component. It is bumped whenever the transformation could produce
	// different bytes for the same inputs, which changes every component
	// identity.
	TransformImplementation = "hachidori.trial-transform/1"

	// RepresentationDense and RepresentationPacked are the representations a
	// group's modules take on the accelerator: the source weights, or the
	// compressed-tensors pack-quantized int4 weights the packed W4A16 forward
	// reads. A future fused/packed execution representation is a new
	// representation value with its own transformation; component identity,
	// Trial and Candidate do not change shape for it.
	RepresentationDense  = "dense"
	RepresentationPacked = "packed-int4"
)

// Member is one module of a component with the material facts of its source
// tensor.
type Member struct {
	Module string  `json:"module"`
	Shape  []int64 `json:"shape"`
	DType  string  `json:"dtype"`
}

// Transformation is the exact, deterministic statement of how a component is
// derived from the source. Backend is what the worker reported for its
// numerical stack (it is material: another compressed-tensors can pack
// differently).
type Transformation struct {
	Policy         string `json:"policy"`
	Representation string `json:"representation"`
	Implementation string `json:"implementation"`
	Backend        string `json:"backend"`
	Scheme         string `json:"scheme,omitempty"`
	Algorithm      string `json:"algorithm,omitempty"`
	Bits           int    `json:"bits,omitempty"`
	GroupSize      int    `json:"group_size,omitempty"`
	Symmetric      bool   `json:"symmetric,omitempty"`
	Format         string `json:"format,omitempty"`
	ComputeDType   string `json:"compute_dtype,omitempty"`
}

// ComponentIdentity is the deterministic identity of one transformed component:
// every input that can change its bytes or the safety of reusing it. It is
// content-derived: no display label, UI state or profile identity is part of it,
// so two profiles that need the same transformation of the same group share the
// component.
type ComponentIdentity struct {
	Schema string `json:"schema"`
	// Source is the exact source model: its repository revision and the digest
	// of every pinned file.
	Source home.VariantSource `json:"source"`
	// TuningSchema is the resolved-plan schema the group identity belongs to.
	TuningSchema string `json:"tuning_schema"`
	// Group is the stable group identity from the analysis, with its region
	// and block.
	Group  string `json:"group"`
	Region string `json:"region"`
	Layer  int    `json:"layer"`
	// Members are the group's modules with their source shape and dtype, in
	// module order.
	Members []Member `json:"members"`
	// Files are the carried source files of a group that has no modules (the
	// joint head). They are never transformed; the source digest pins them.
	Files          []string       `json:"files,omitempty"`
	Transformation Transformation `json:"transformation"`
}

// Canonical is the identity's canonical encoding.
func (c ComponentIdentity) Canonical() []byte {
	b, err := json.Marshal(c)
	if err != nil {
		panic(err)
	}
	return b
}

// ID is the component identity digest.
func (c ComponentIdentity) ID() string { return sha256Hex(c.Canonical()) }

// Validate checks the identity names everything it must.
func (c ComponentIdentity) Validate() error {
	switch {
	case c.Schema != ComponentSchema:
		return fmt.Errorf("component schema %q, want %q", c.Schema, ComponentSchema)
	case c.Source.ID == "" || c.Source.Revision == "" || c.Source.FilesSHA256 == "":
		return errors.New("component identity does not name its exact source")
	case c.TuningSchema == "" || c.Group == "" || c.Region == "":
		return errors.New("component identity does not name its tuning schema and group")
	case len(c.Members)+len(c.Files) == 0:
		return fmt.Errorf("component %s has no members", c.Group)
	case len(c.Members) == 0 && c.Transformation.Policy != home.PolicySourcePrecision:
		return fmt.Errorf("component %s carries only files and cannot be transformed", c.Group)
	case c.Transformation.Policy == "" || c.Transformation.Representation == "" || c.Transformation.Implementation == "":
		return errors.New("component identity does not name its transformation")
	}
	for i, m := range c.Members {
		if m.Module == "" || len(m.Shape) == 0 || m.DType == "" {
			return fmt.Errorf("component %s member %q lacks a module, shape or dtype", c.Group, m.Module)
		}
		if i > 0 && c.Members[i-1].Module >= m.Module {
			return fmt.Errorf("component %s members are not in module order", c.Group)
		}
	}
	return nil
}

// Contract resolves the transformation a policy denotes for the canonical
// recipe of a plan. It reads the scheme's parameters from the same declaration
// the Forge builder verifies a variant against (optimize.WeightsOf), so a trial
// and a build cannot silently disagree about them. A policy the optimizer does
// not execute has no contract.
func Contract(source home.VariantSource, plan home.TuningPlan, policy, backend string) (Transformation, error) {
	switch policy {
	case home.PolicySourcePrecision:
		return Transformation{Policy: policy, Representation: RepresentationDense, Implementation: TransformImplementation, Backend: backend}, nil
	case home.PolicyW4A16:
		recipe, err := optimize.LookupRecipe(source.ID, plan.Recipe)
		if err != nil {
			return Transformation{}, err
		}
		w, err := optimize.WeightsOf(recipe)
		if err != nil {
			return Transformation{}, err
		}
		if w.Format != "compressed-tensors/pack-quantized" || w.Bits != 4 || w.GroupSize <= 0 || !w.Symmetric {
			return Transformation{}, fmt.Errorf("trial transformation for scheme %s is not supported", w.Scheme)
		}
		return Transformation{Policy: policy, Representation: RepresentationPacked, Implementation: TransformImplementation, Backend: backend,
			Scheme: w.Scheme, Algorithm: recipe.Algorithm, Bits: w.Bits, GroupSize: w.GroupSize, Symmetric: w.Symmetric, Format: w.Format, ComputeDType: w.DType}, nil
	}
	return Transformation{}, fmt.Errorf("policy %q is not a transformation the optimizer executes", policy)
}

// Identity builds the identity of one plan group's component under policy.
// members are the modules' source tensor facts as the worker observed them;
// every module of the group must be among them.
func Identity(source home.VariantSource, plan home.TuningPlan, group home.TuningGroupPlan, policy, backend string, observed map[string]Member) (ComponentIdentity, error) {
	t, err := Contract(source, plan, policy, backend)
	if err != nil {
		return ComponentIdentity{}, err
	}
	modules := append([]string(nil), group.Modules...)
	sort.Strings(modules)
	members := make([]Member, 0, len(modules))
	for _, m := range modules {
		o, ok := observed[m]
		if !ok {
			return ComponentIdentity{}, fmt.Errorf("group %s names module %s that the resident source does not have", group.ID, m)
		}
		members = append(members, Member{Module: m, Shape: append([]int64(nil), o.Shape...), DType: o.DType})
	}
	files := append([]string(nil), group.Files...)
	sort.Strings(files)
	id := ComponentIdentity{Schema: ComponentSchema, Source: source, TuningSchema: plan.Schema, Group: group.ID, Region: group.Region,
		Layer: group.Layer, Members: members, Files: files, Transformation: t}
	return id, id.Validate()
}

// ComponentBytes is the exact size in bytes the component of a group occupies
// in system RAM: derived from the members' tensor geometry and the
// transformation, never estimated from a measurement. The worker reports the
// actual size after building it and a difference is a failure.
func ComponentBytes(c ComponentIdentity) (int64, error) {
	if c.Transformation.Representation == RepresentationDense {
		// Source-precision material is the canonical source itself.
		return 0, nil
	}
	if c.Transformation.Representation != RepresentationPacked {
		return 0, fmt.Errorf("unknown representation %q", c.Transformation.Representation)
	}
	scale, ok := dtypeBytes(c.Transformation.ComputeDType)
	if !ok {
		return 0, fmt.Errorf("unknown compute dtype %q", c.Transformation.ComputeDType)
	}
	t := c.Transformation
	var total int64
	for _, m := range c.Members {
		if len(m.Shape) != 2 {
			return 0, fmt.Errorf("module %s is not a matrix (shape %v)", m.Module, m.Shape)
		}
		out, in := m.Shape[0], m.Shape[1]
		if t.GroupSize <= 0 || in%int64(t.GroupSize) != 0 {
			return 0, fmt.Errorf("module %s width %d is not a multiple of group size %d", m.Module, in, t.GroupSize)
		}
		words := (in*int64(t.Bits) + 31) / 32
		total += out*words*4 + out*(in/int64(t.GroupSize))*scale + 2*8 // packed int32, scales, weight_shape (two int64)
	}
	return total, nil
}

func dtypeBytes(name string) (int64, bool) {
	switch name {
	case "bfloat16", "float16":
		return 2, true
	case "float32":
		return 4, true
	}
	return 0, false
}

// SourceBytes is the size of a group's source tensors.
func SourceBytes(members []Member) (int64, error) {
	var total int64
	for _, m := range members {
		n, ok := dtypeBytes(m.DType)
		if !ok {
			return 0, fmt.Errorf("module %s has unsupported dtype %q", m.Module, m.DType)
		}
		for _, d := range m.Shape {
			n *= d
		}
		total += n
	}
	return total, nil
}

func sha256Hex(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}
