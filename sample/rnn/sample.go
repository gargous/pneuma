package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
)

func novelProc(dstPath, srcPath string) error {
	os.Mkdir(dstPath, os.ModePerm)
	dirs, err := os.ReadDir(srcPath)
	if err != nil {
		return err
	}
	for _, dir := range dirs {
		data, err := os.ReadFile(filepath.Join(srcPath, dir.Name()))
		if err != nil {
			return err
		}
		fmt.Println(dir.Name(), len(data))
		data = bytes.ReplaceAll(data, []byte{' '}, []byte{})
		data = bytes.ReplaceAll(data, []byte{'\t'}, []byte{})
		phases := bytes.Split(data, []byte{'\n'})
		for i := 0; i < len(phases); {
			if len(phases[i]) == 0 || (len(phases[i]) == 1 && phases[i][0] == '\r') {
				phases = append(phases[:i], phases[i+1:]...)
			} else {
				i++
			}
		}
		fmt.Println(dir.Name(), len(data), len(phases))
		data = bytes.Join(phases, []byte{'\n'})
		err = os.WriteFile(filepath.Join(dstPath, dir.Name()), data, os.ModePerm)
		if err != nil {
			return err
		}
	}
	return nil
}
