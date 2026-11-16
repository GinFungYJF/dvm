/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _DVM_PY_API_H_
#define _DVM_PY_API_H_
#include <memory>
#include <string>
#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "dvm.h"

namespace dvm {
namespace py = pybind11;
class DVM_API DataTypePy {
 public:
  explicit constexpr DataTypePy(DataType dtype) : dtype_(dtype) {}
  operator DataType() const { return dtype_; }
  DataType dtype_;
};

class DVM_API NDObjectPy {
 public:
  explicit NDObjectPy(NDObject *obj) : obj_(obj) {}
  py::object GetShape() const {
    auto *shape_ref = Kernel::GetShape(obj_);
    const size_t size = shape_ref->size;
    py::tuple out(size);
    for (size_t i = 0; i < size; ++i) {
      out[i] = py::cast(shape_ref->data[i]);
    }
    return out;
  }
  DataTypePy GetDType() const { return DataTypePy(Kernel::GetDType(obj_)); }
  NDObject *Get() const { return obj_; }

 private:
  NDObject *obj_;
};

class DVM_API IntArrayRefPy {
 public:
  IntArrayRefPy() : shape_ref_(shape_) {}
  explicit IntArrayRefPy(const std::vector<int64_t> &shape) : shape_(shape), shape_ref_(shape_) {}
  ~IntArrayRefPy() = default;
  void Update(py::object shape) {
    shape_ = py::cast<std::vector<int64_t>>(shape);
    shape_ref_ = shape_;
  }
  py::object GetShape() const {
    const size_t size = shape_.size();
    py::tuple out(size);
    for (size_t i = 0; i < size; ++i) {
      out[i] = py::cast(shape_[i]);
    }
    return out;
  }
  IntArrayRef *Get() { return &shape_ref_; }

 private:
  std::vector<int64_t> shape_;
  IntArrayRef shape_ref_;
};

class DVM_API ScalarRefPy {
 public:
  ScalarRefPy() { data_.type = kDataTypeEnd; }
  explicit ScalarRefPy(DataTypePy type) { data_.type = type; }
  void Update(py::object val) {
    if (py::isinstance<py::int_>(val)) {
      data_ = val.cast<int64_t>();
    } else if (py::isinstance<py::float_>(val)) {
      data_ = val.cast<float>();
    } else {
      DvmException("Unsupported scalar value type for current data type.");
    }
  }
  ScalarRef data_;
};

using NDOpPyPtr = std::shared_ptr<NDObjectPy>;
using ScalarRefPyPtr = std::shared_ptr<ScalarRefPy>;

class DVM_API KernelPy {
 public:
  KernelPy() = default;
  virtual ~KernelPy() {}

  virtual py::object Load(py::object shape, DataTypePy type) = 0;
  virtual py::object GlobalAccess(py::object shape, DataTypePy type) = 0;
  virtual py::object ViewLoad(py::object shape, py::object stride, DataTypePy type) = 0;
  virtual py::object GatherLoad(py::object shape, py::object index, DataTypePy type, int axis, int gather_mode) = 0;
  virtual py::object Store(py::object obj, DataTypePy type) = 0;
  virtual py::object ViewStore(py::object obj, py::object stride, DataTypePy type) = 0;

  template <UnaryOpType op_type>
  py::object Unary(py::object input) {
    return ObjToPy(kernel_.Unary<op_type>(PyToObj(input)));
  }

  template <BinaryOpType op_type>
  py::object Binary(py::object lhs, py::object rhs) {
    NDObject *op;
    if (py::isinstance<py::int_>(lhs)) {
      op = kernel_.Binary<op_type>(lhs.cast<int64_t>(), PyToObj(rhs));
    } else if (py::isinstance<py::float_>(lhs)) {
      op = kernel_.Binary<op_type>(lhs.cast<float>(), PyToObj(rhs));
    } else if (py::isinstance<py::int_>(rhs)) {
      op = kernel_.Binary<op_type>(PyToObj(lhs), rhs.cast<int64_t>());
    } else if (py::isinstance<py::float_>(rhs)) {
      op = kernel_.Binary<op_type>(PyToObj(lhs), rhs.cast<float>());
    } else if (py::isinstance<ScalarRefPy>(lhs)) {
      op = kernel_.Binary<op_type>(PyToScalar(lhs), PyToObj(rhs));
    } else if (py::isinstance<ScalarRefPy>(rhs)) {
      op = kernel_.Binary<op_type>(PyToObj(lhs), PyToScalar(rhs));
    } else {
      op = kernel_.Binary<op_type>(PyToObj(lhs), PyToObj(rhs));
    }
    return ObjToPy(op);
  }

  template <ReduceOpType op_type>
  py::object Reduce(py::object input, py::object dims, bool keepdims) {
    return ObjToPy(kernel_.Reduce<op_type>(PyToObj(input), GetShapeRef(dims), keepdims));
  }

  py::object Cast(py::object input, DataTypePy type) { return ObjToPy(kernel_.Cast(PyToObj(input), type)); }
  py::object Select(py::object cond, py::object lhs, py::object rhs) {
    return ObjToPy(kernel_.Select(PyToObj(cond), PyToObj(lhs), PyToObj(rhs)));
  }
  py::object Full(py::object scalar, py::object shape, DataTypePy dtype) {
    auto shape_ref = GetShapeRef(shape);
    NDObject *op = nullptr;
    if (py::isinstance<py::bool_>(scalar)) {
      op = kernel_.Broadcast(static_cast<int>(scalar.cast<bool>()), shape_ref, dtype);
    } else if (py::isinstance<py::int_>(scalar)) {
      op = kernel_.Broadcast(scalar.cast<int64_t>(), shape_ref, dtype);
    } else if (py::isinstance<py::float_>(scalar)) {
      op = kernel_.Broadcast(scalar.cast<float>(), shape_ref, dtype);
    } else if (py::isinstance<ScalarRefPy>(scalar)) {
      op = kernel_.Broadcast(PyToScalar(scalar), shape_ref, dtype);
    } else {
      DvmException("Unsupported scalar type for full: expected bool, int, float, or ScalarRef.");
    }
    return ObjToPy(op);
  }
  py::object Reshape(py::object input, py::object shape) {
    return ObjToPy(kernel_.Reshape(PyToObj(input), GetShapeRef(shape)));
  }
  py::object Permute(py::object input, py::object dims) {
    return ObjToPy(kernel_.Permute(PyToObj(input), GetShapeRef(dims)));
  }
  py::object Slice(py::object input, py::object start, py::object size) {
    return ObjToPy(kernel_.Slice(PyToObj(input), GetShapeRef(start), GetShapeRef(size)));
  }
  py::object SliceDim(py::object input, int dim, py::object begin, py::object end) {
    return ObjToPy(kernel_.Slice(PyToObj(input), dim, GetScalarRef(begin), GetScalarRef(end)));
  }
  py::object Copy(py::object input) { return ObjToPy(kernel_.Copy(PyToObj(input))); }
  py::object Concat(py::sequence inputs, int dim) {
    std::vector<NDObject *> objs;
    size_t size = py::len(inputs);
    objs.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      objs.push_back(PyToObj(inputs[i].cast<py::object>()));
    }
    return ObjToPy(kernel_.Concat(objs.data(), objs.size(), dim));
  }
  py::list Split(py::object input, int dim, int64_t split_size, size_t split_num) {
    auto objs = kernel_.Split(PyToObj(input), dim, split_size, split_num);
    py::tuple out(split_num);
    for (size_t i = 0; i < split_num; ++i) {
      out[i] = ObjToPy(objs[i]);
    }
    return out;
  }
  py::object Broadcast(py::object input, py::object shape) {
    return ObjToPy(kernel_.Broadcast(PyToObj(input), GetShapeRef(shape)));
  }
  py::object ElementAny(py::object input) { return ObjToPy(kernel_.ElemAny(PyToObj(input))); }
  py::object MatMul(py::object lhs, py::object rhs, bool trans_a, bool trans_b, py::object bias) {
    auto op = kernel_.MatMul(PyToObj(lhs), PyToObj(rhs), trans_a, trans_b, bias.is_none() ? nullptr : PyToObj(bias));
    return ObjToPy(op);
  }
  py::object GroupedMatMul(py::object lhs, py::object rhs, bool trans_a, bool trans_b, py::object bias,
                           py::object group_list, int64_t group_type, int64_t group_list_type) {
    NDObject *bias_obj = bias.is_none() ? nullptr : PyToObj(bias);
    NDObject *group_list_obj = group_list.is_none() ? nullptr : PyToObj(group_list);
    auto op = kernel_.GroupedMatMul(PyToObj(lhs), PyToObj(rhs), trans_a, trans_b, bias_obj, group_list_obj,
                                    GroupType(group_type), GroupListType(group_list_type));
    return ObjToPy(op);
  }
  void SetStoreInplace(py::object store) { kernel_.SetStoreInplace(PyToObj(store)); }
  void SetStoreTemp(py::object store) { kernel_.SetStoreTemp(PyToObj(store)); }
  void SetLoadBind(py::object load, py::object access) { kernel_.SetLoadBind(PyToObj(load), PyToObj(access)); }
  py::object DisAssemble() { return py::cast(kernel_.Das()); }
  py::object DumpGraph() { return py::cast(kernel_.Dump()); }
  py::object DumpR() { return py::cast(kernel_.DumpR()); }
  py::object Inspect() { return py::cast(kernel_.Inspect()); }
  void SpecNext() { kernel_.SpecNext(); }
  void ParallelAdd(int ktype, uint32_t flags, int core_limit) { kernel_.ParallelAdd(static_cast<KernelType>(ktype), flags, core_limit); }
  void SequenceAdd(int ktype, uint32_t flags) { kernel_.SequenceAdd(static_cast<KernelType>(ktype), flags); }
  py::object MakeIntArray() { return py::cast(std::make_shared<IntArrayRefPy>()); }
  py::object MakeScalar(DataTypePy type) { return py::cast(std::make_shared<ScalarRefPy>(type)); }

  static void SetDeterm(bool enable) {
    if (enable) {
      Config::Instance().SetDeterm();
    } else {
      Config::Instance().UnsetDeterm();
    }
  }
  static void SetOnlineTuning(bool enable) {
    if (enable) {
      Config::Instance().SetOnlineTuner();
    } else {
      Config::Instance().UnsetOnlineTuner();
    }
  }
  static void SetVfFusion(int mode) { Config::Instance().SetVfFusion(mode); }

  static constexpr int K_VEC = KernelType::kVector;
  static constexpr int K_CUBE = KernelType::kCube;
  static constexpr int K_MIX = KernelType::kMix;
  static constexpr int K_PARAL = KernelType::kParallel;
  static constexpr int K_SEQ = KernelType::kSequence;
  static constexpr int K_SPLIT = KernelType::kSplit;
  static constexpr int K_EAGER = KernelType::kEager;
  static constexpr uint32_t F_DYN = KernelFlag::kDynamic;
  static constexpr uint32_t F_UWS = KernelFlag::kUnifyWS;
  static constexpr uint32_t F_SPEC = KernelFlag::kSpeculate;
  static constexpr uint32_t F_OPT_FRAC = KernelFlag::kOptFractalTrans;
  static constexpr uint32_t F_PRIV1 = KernelFlag::kPrivate1;

 protected:
  virtual IntArrayRef *GetShapeRef(py::object shape) = 0;
  NDObject *PyToObj(py::object obj) { return obj.cast<NDOpPyPtr>()->Get(); }
  py::object ObjToPy(NDObject *obj) { return py::cast(std::make_shared<NDObjectPy>(obj)); }
  ScalarRef *PyToScalar(py::object scalar) { return &(scalar.cast<ScalarRefPyPtr>()->data_); }
  ScalarRef *GetScalarRef(py::object scalar) {
    if (py::isinstance<ScalarRefPy>(scalar)) {
      return PyToScalar(scalar);
    }
    if (!py::isinstance<py::int_>(scalar)) {
      DvmException("Unsupported slice_dim bound type: expected int or ScalarRef (used for dynamic scalar inputs).");
      return nullptr;
    }
    auto ref = std::make_shared<ScalarRefPy>(DataTypePy(kInt64));
    ref->Update(scalar);
    owned_scalars_.push_back(ref);
    return &(ref->data_);
  }
  Kernel kernel_;
  std::vector<ScalarRefPyPtr> owned_scalars_;
};

constexpr auto bool_py = DataTypePy(kBool);
constexpr auto float16_py = DataTypePy(kFloat16);
constexpr auto bfloat16_py = DataTypePy(kBFloat16);
constexpr auto float32_py = DataTypePy(kFloat32);
constexpr auto int32_py = DataTypePy(kInt32);
constexpr auto int64_py = DataTypePy(kInt64);

static inline void RegDvmPy(const py::module &m) {
  (void)py::class_<NDObjectPy, std::shared_ptr<NDObjectPy>>(m, "NDObject")
    .def("shape", &NDObjectPy::GetShape, "get shape")
    .def("dtype", &NDObjectPy::GetDType, "get dtype");

  (void)py::class_<DataTypePy>(m, "DataType")
    .def("__eq__", [](const DataTypePy &self, const DataTypePy &other) { return self.dtype_ == other.dtype_; })
    .def("__ne__", [](const DataTypePy &self, const DataTypePy &other) { return self.dtype_ != other.dtype_; })
    .def_readonly_static("bool", &bool_py)
    .def_readonly_static("float16", &float16_py)
    .def_readonly_static("bfloat16", &bfloat16_py)
    .def_readonly_static("float32", &float32_py)
    .def_readonly_static("int32", &int32_py)
    .def_readonly_static("int64", &int64_py);

  (void)py::class_<IntArrayRefPy, std::shared_ptr<IntArrayRefPy>>(m, "IntArrayRef")
    .def(py::init<>())
    .def(py::init<const std::vector<int64_t> &>())
    .def("shape", &IntArrayRefPy::GetShape, "get shape")
    .def("update", &IntArrayRefPy::Update, "update shape");

  (void)py::class_<ScalarRefPy, std::shared_ptr<ScalarRefPy>>(m, "ScalarRef")
    .def("update", &ScalarRefPy::Update, "update value");

  (void)py::class_<KernelPy, std::shared_ptr<KernelPy>>(m, "Kernel")
    .def("load", &KernelPy::Load, "load array")
    .def("global_access", &KernelPy::GlobalAccess, "create global access")
    .def("view_load", &KernelPy::ViewLoad, "load array")
    .def("gather_load", &KernelPy::GatherLoad, "gather load array", py::arg("shape"), py::arg("index"),
         py::arg("type"), py::arg("axis") = 0, py::arg("gather_mode") = static_cast<int>(kElementGather))
    .def("store", &KernelPy::Store, "store array", py::arg("obj"), py::arg("type") = DataTypePy(kDataTypeEnd))
    .def("view_store", &KernelPy::ViewStore, "store array with stride", py::arg("obj"), py::arg("stride"),
         py::arg("type") = DataTypePy(kDataTypeEnd))
    .def("set_store_inplace", &KernelPy::SetStoreInplace, "store inplace")
    .def("set_store_temp", &KernelPy::SetStoreTemp, "mark store as temporary")
    .def("set_load_bind", &KernelPy::SetLoadBind, "mark load bind")
    .def("scalar", &KernelPy::MakeScalar, "create scalar", py::arg("dtype") = DataTypePy(kDataTypeEnd))
    .def("int_array", &KernelPy::MakeIntArray, "create int array")
    .def("sqrt", &KernelPy::Unary<UnaryOpType::kSqrt>, "emit sqrt")
    .def("abs", &KernelPy::Unary<UnaryOpType::kAbs>, "emit abs")
    .def("log", &KernelPy::Unary<UnaryOpType::kLog>, "emit log")
    .def("exp", &KernelPy::Unary<UnaryOpType::kExp>, "emit exp")
    .def("reciprocal", &KernelPy::Unary<UnaryOpType::kReciprocal>, "emit reciprocal")
    .def("isfinite", &KernelPy::Unary<UnaryOpType::kIsFinite>, "emit isfinite")
    .def("logical_not", &KernelPy::Unary<UnaryOpType::kLogicalNot>, "emit logical_not")
    .def("round", &KernelPy::Unary<UnaryOpType::kRound>, "emit round")
    .def("floor", &KernelPy::Unary<UnaryOpType::kFloor>, "emit floor")
    .def("ceil", &KernelPy::Unary<UnaryOpType::kCeil>, "emit ceil")
    .def("trunc", &KernelPy::Unary<UnaryOpType::kTrunc>, "emit trunc")
    .def("cast", &KernelPy::Cast, "emit cast op")
    .def("element_any", &KernelPy::ElementAny, "emit element_any op")
    .def("equal", &KernelPy::Binary<BinaryOpType::kEqual>, "emit equal")
    .def("not_equal", &KernelPy::Binary<BinaryOpType::kNotEqual>, "emit equal")
    .def("greater", &KernelPy::Binary<BinaryOpType::kGreater>, "emit greater")
    .def("greater_equal", &KernelPy::Binary<BinaryOpType::kGreaterEqual>, "emit greater_equal")
    .def("less", &KernelPy::Binary<BinaryOpType::kLess>, "emit less")
    .def("less_equal", &KernelPy::Binary<BinaryOpType::kLessEqual>, "emit less_equal")
    .def("add", &KernelPy::Binary<BinaryOpType::kAdd>, "emit add")
    .def("sub", &KernelPy::Binary<BinaryOpType::kSub>, "emit sub")
    .def("mul", &KernelPy::Binary<BinaryOpType::kMul>, "emit mul")
    .def("div", &KernelPy::Binary<BinaryOpType::kDiv>, "emit div")
    .def("pow", &KernelPy::Binary<BinaryOpType::kPow>, "emit pow")
    .def("maximum", &KernelPy::Binary<BinaryOpType::kMaximum>, "emit maximum")
    .def("minimum", &KernelPy::Binary<BinaryOpType::kMinimum>, "emit minimum")
    .def("logical_and", &KernelPy::Binary<BinaryOpType::kLogicalAnd>, "emit logical_add")
    .def("logical_or", &KernelPy::Binary<BinaryOpType::kLogicalOr>, "emit logical_or")
    .def("select", &KernelPy::Select, "emit select op")
    .def("broadcast", &KernelPy::Broadcast, "emit broadcast op")
    .def("full", &KernelPy::Full, "emit full op")
    .def("reshape", &KernelPy::Reshape, "emit reshape op")
    .def("permute", &KernelPy::Permute, "emit permute op")
    .def("slice", &KernelPy::Slice, "emit slice op")
    .def("slice_dim", &KernelPy::SliceDim, "emit dim slice op")
    .def("sum", &KernelPy::Reduce<ReduceOpType::kSum>, py::arg("input"), py::arg("dims"), py::arg("keepdims") = false,
         "emit sum")
    .def("max", &KernelPy::Reduce<ReduceOpType::kMax>, py::arg("input"), py::arg("dims"), py::arg("keepdims") = false,
         "emit max")
    .def("min", &KernelPy::Reduce<ReduceOpType::kMin>, py::arg("input"), py::arg("dims"), py::arg("keepdims") = false,
         "emit min")
    .def("copy", &KernelPy::Copy, "emit copy op")
    .def("concat", &KernelPy::Concat, "emit concat op")
    .def("split", &KernelPy::Split, "emit split op")
    .def("matmul", &KernelPy::MatMul, "emit matmul op", py::arg("lhs"), py::arg("rhs"), py::arg("trans_a"),
         py::arg("trans_b"), py::arg("bias") = py::none())
    .def("grouped_matmul", &KernelPy::GroupedMatMul, "emit grouped_matmul op", py::arg("lhs"), py::arg("rhs"),
         py::arg("trans_a"), py::arg("trans_b"), py::arg("bias"), py::arg("group_list"), py::arg("group_type"),
         py::arg("group_list_type") = 0)
    .def("das", &KernelPy::DisAssemble, "disassemble code")
    .def("dump", &KernelPy::DumpGraph, "dump graph")
    .def("dump_r", &KernelPy::DumpR, "dump ref graph")
    .def("inspect", &KernelPy::Inspect, "inspect for health")
    .def("spec_next", &KernelPy::SpecNext, "spec next")
    .def("parallel_add", &KernelPy::ParallelAdd, "add new parallel Kernel", py::arg("ktype"), py::arg("flags") = 0,
         py::arg("core_limit") = 0)
    .def("seq_add", &KernelPy::SequenceAdd, "add new sequence Kernel", py::arg("ktype"), py::arg("flags") = 0)
    .def_readonly_static("K_VEC", &KernelPy::K_VEC)
    .def_readonly_static("K_CUBE", &KernelPy::K_CUBE)
    .def_readonly_static("K_MIX", &KernelPy::K_MIX)
    .def_readonly_static("K_PARAL", &KernelPy::K_PARAL)
    .def_readonly_static("K_SEQ", &KernelPy::K_SEQ)
    .def_readonly_static("K_SPLIT", &KernelPy::K_SPLIT)
    .def_readonly_static("K_EAGER", &KernelPy::K_EAGER)
    .def_readonly_static("F_DYN", &KernelPy::F_DYN)
    .def_readonly_static("F_UWS", &KernelPy::F_UWS)
    .def_readonly_static("F_SPEC", &KernelPy::F_SPEC)
    .def_readonly_static("F_OPT_FRAC", &KernelPy::F_OPT_FRAC)
    .def_readonly_static("F_PRIV1", &KernelPy::F_PRIV1)
    .def_static("set_deterministic", &KernelPy::SetDeterm, "set deterministic")
    .def_static("set_online_tuning", &KernelPy::SetOnlineTuning, "set online tuning")
    .def_static("set_vf_fusion", &KernelPy::SetVfFusion,
                "set C310 VF fusion mode: 0 disables; 1 validates shapes; 2 assumes equal shapes", py::arg("mode"));
}
}  // namespace dvm
#endif  // _DVM_PY_API_H_
