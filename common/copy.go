package common

import (
	"reflect"
)

func Copy(src interface{}) interface{} {
	srcVal := reflect.ValueOf(src)
	srcTyp := reflect.TypeOf(src).Elem()
	dstVal := reflect.New(srcTyp)
	dstVal.MethodByName("Copy").Call([]reflect.Value{srcVal})
	return dstVal.Interface()
}

func CopyIHLayer(src IHLayer) IHLayer {
	return Copy(src).(IHLayer)
}

func CopyITarget(src ITarget) ITarget {
	return Copy(src).(ITarget)
}

func CopyIOptimizer(src IOptimizer) IOptimizer {
	return Copy(src).(IOptimizer)
}
