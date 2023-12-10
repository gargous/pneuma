package common

import "reflect"

func Copy(src interface{}) interface{} {
	srcVal := reflect.ValueOf(src)
	srcTyp := reflect.TypeOf(src)
	dstVal := reflect.New(srcTyp)
	dstVal.MethodByName("Copy").Call([]reflect.Value{srcVal})
	return dstVal.Interface()
}

func CopyIHLayer(srcHL IHLayer) IHLayer {
	return Copy(srcHL).(IHLayer)
}

func CopyITarget(src ITarget) ITarget {
	return Copy(src).(ITarget)
}

func CopyIOptimizer(src IOptimizer) IOptimizer {
	return Copy(src).(IOptimizer)
}
