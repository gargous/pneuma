package main

import "path/filepath"

func main() {
	if err := novelProc(filepath.Join("./resource", "novel_fix"), filepath.Join("./resource", "novel")); err != nil {
		panic(err)
	}
}
