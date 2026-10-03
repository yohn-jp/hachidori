package dashboard

type resourceChoice struct {
	Label      string
	FormAction string
	Name       string
	Value      string
}

// resourceSelection is the shared dashboard presentation for choosing one
// resource or entering its exact path when a native picker is unavailable.
type resourceSelection struct {
	Native      bool
	Label       string
	Name        string
	Value       string
	Placeholder string
	Required    bool
	Multiline   bool
	Choices     []resourceChoice
}

func resourceInput(native bool, label, name, value, placeholder string, required, multiline bool, formAction, choiceName string, choices ...string) resourceSelection {
	s := resourceSelection{
		Native: native, Label: label, Name: name, Value: value,
		Placeholder: placeholder, Required: required, Multiline: multiline,
	}
	for i := 0; i+1 < len(choices); i += 2 {
		s.Choices = append(s.Choices, resourceChoice{
			Label: choices[i+1], FormAction: formAction, Name: choiceName, Value: choices[i],
		})
	}
	return s
}
