package setup

// PythonArgs is the interpreter command line of every private Python process
// that can import modules from an immutable source or variant artifact tree:
// isolated, UTF-8, and with bytecode writes disabled (-B) so an imported
// artifact module can never leave __pycache__ beside the artifact's files.
// The flag is required beside PYTHONDONTWRITEBYTECODE in home.Home.Env:
// isolated mode (-I) makes the interpreter ignore every PYTHON* variable, so
// the environment alone does not protect the artifact. rest is the script and
// its arguments.
func PythonArgs(rest ...string) []string {
	return append([]string{"-I", "-B", "-X", "utf8"}, rest...)
}
