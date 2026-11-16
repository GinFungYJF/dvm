/**
 * Copyright 2024-2026 Huawei Technologies Co., Ltd
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

#ifndef _DVM_H_
#define _DVM_H_

#include <cstdint>
#include <cstddef>
#include <vector>

#define DVM_API __attribute__((visibility("default")))

namespace dvm {
enum DataType {
  kBool = 0,
  kFloat16,
  kBFloat16,
  kFloat32,
  kInt32,
  kInt64,
  kDataTypeEnd,
};

enum GatherMode {
  kElementGather = 0,
  kSliceGather = 1,
};

enum UnaryType {
  kSqrt = 0,
  kAbs,
  kLog,
  kExp,
  kReciprocal,
  kIsFinite,
  kLogicalNot,
  kRound,
  kFloor,
  kCeil,
  kTrunc,
  kUnaryTypeEnd,
};

enum BinaryType {
  kEqual = 0,
  kNotEqual,
  kGreater,
  kGreaterEqual,
  kLess,
  kLessEqual,
  kAdd,
  kSub,
  kMul,
  kDiv,
  kPow,
  kMaximum,
  kMinimum,
  kLogicalAnd,
  kLogicalOr,
  kBinaryTypeEnd,
};

enum ReduceType {
  kSum = 0,
  kMax,
  kMin,
  kReduceTypeEnd,
};

enum GmmSplitType {
  kSplit_M = 0,
  kSplit_N,
  kSplit_K,
  kGroupTypeEnd,
};

enum GmmListType {
  kListBound = 0,
  kListSize,
  kEnd,
};

/** kernel type for reset. */
enum KernelType {
  kVector = 0,    /**< kernel with only vector operations */
  kCube,          /**< kernel with only cube operations */
  kMix,           /**< kernel with one cube operation and some post-fusioned vector operations */
  kParallel,      /**< kernel with multi sub-kernels parallel processed on diffrent cores */
  kSequence,      /**< kernel with multi sub-kernels sequence processed */
  kSplit,         /**< kernel with auto sub-kernel split */
  kEager,         /**< kernel with eager constructing operations  */
  kKernelTypeEnd, /**< kernel end */
};

/** kernel flags for reset. */
enum KernelFlag {
  kDynamic = 0x1,
  kUnifyWS = 0x2,
  kSpeculate = 0x4,
  kLazyCodeGen = 0x8,
  kOptFractalTrans = 1u << 29,
  kPrivate1 = 1u << 30,
  kPrivate2 = 1u << 31,
};

class NDObject;
class VKernel;
class Communicator;

class DVM_API Float16 {
 public:
  explicit Float16(uint16_t v) : value_(v) {}
  explicit Float16(float v);
  explicit Float16(int32_t v);
  explicit operator float() const;
  explicit operator int32_t() const;
  uint16_t int_value() const { return value_; }

 private:
  uint16_t value_;
};

class DVM_API BFloat16 {
 public:
  explicit BFloat16(uint16_t v) : value_(v) {}
  explicit BFloat16(float v);
  explicit BFloat16(int32_t v);
  explicit operator float() const;
  explicit operator int32_t() const;
  uint16_t int_value() const { return value_; }

 private:
  uint16_t value_;
};

struct DVM_API IntArrayRef {
  IntArrayRef() {}
  explicit IntArrayRef(const std::vector<int64_t> &other) : data(other.data()), size(other.size()) {}
  IntArrayRef &operator=(const std::vector<int64_t> &other) {
    data = other.data();
    size = other.size();
    return *this;
  }
  const int64_t *data;
  size_t size;
};

struct DVM_API ScalarRef {
  ScalarRef() = default;
  template <typename T>
  ScalarRef(T val) {
    *this = val;
  }
  ScalarRef &operator=(float val) {
    type = kFloat32;
    f32 = val;
    return *this;
  }
  ScalarRef &operator=(int32_t val) {
    type = kInt32;
    i32 = val;
    return *this;
  }
  ScalarRef &operator=(Float16 val) {
    type = kFloat16;
    f16 = val.int_value();
    return *this;
  }
  ScalarRef &operator=(BFloat16 val) {
    type = kBFloat16;
    f16 = val.int_value();
    return *this;
  }
  ScalarRef &operator=(int64_t val) {
    type = kInt64;
    i64 = val;
    return *this;
  }
  DataType type;
  union {
    float f32;
    int32_t i32;
    uint16_t f16;
    int64_t i64;
  };
};

struct DVM_API RelocEntry {
  RelocEntry() {}
  RelocEntry(NDObject *p, void *a) : io(p), addr(a) {}
  NDObject *io;
  void *addr;
};

struct DVM_API WsAllocator {
  virtual void *Alloc(size_t size) = 0;
};

struct DVM_API CloneHelper {
  virtual IntArrayRef *GetClone(IntArrayRef *shape) = 0;
  virtual ScalarRef *GetClone(ScalarRef *scalar) = 0;
  virtual NDObject *GetClone(NDObject *op) = 0;
  virtual void SetClone(NDObject *op, NDObject *clone) = 0;
};

class DVM_API Comm {
 public:
  enum { kMemory, kHccl, kDummy };
  Comm() = default;
  ~Comm();
  bool Init(int rank_id, int rank_size, int comm_type, const uint32_t *group_ranks = nullptr);
  bool Init(int rank_id, int rank_size) { return Init(rank_id, rank_size, kMemory); }
  void Init(void *hccl_comm);
  inline const Communicator *GetImpl() const { return comm_; }

 protected:
  Communicator *comm_{nullptr};
};

/**
 * @brief The main Kernel class.
 */
class DVM_API Kernel {
 public:
  Kernel();
  ~Kernel();

  /**
   * @brief Kernel copy. ONLY one Kernel hold the implementation instance.
   */
  Kernel& operator=(Kernel &k) {
    kernel_ = k.kernel_;
    k.kernel_ = nullptr;
    return *this;
  }
  explicit Kernel(Kernel &k) { *this = k; }

  /**
   * @brief Set this kernel to a specific kernel type.
   * @param type target kernel type.
   * @param flags flag options combined of KernelFlag::XX.
   */
  void Reset(KernelType type, uint32_t flags);

  /**
   * @brief Clone kernel represent from a base kernel.
   * @param base the base kernel to clone. MUST BE CONSTRUCTED!
   * @param helper clone callback helper. used to manage object correspondence.
   */
  void Clone(const Kernel &base, CloneHelper &helper);

  /**
   * @brief Set name hint for this kernel. used for msprof, dump etc.
   * @param name short category name. like: "AddSum"
   * @param fullname full name. same as name or with more special info, like: "attention/AddSum"
   */
  void SetNameHint(const char *name, const char *fullname);

  /**
   * @brief Emit a continuous load operation from input tensor.
   * @param addr memory address of input tensor. also can be relocated at codegen stage.
   * @param shape shape reference of input tensor.
   * @param dtype data type of input tensor.
   * @return the result load operation.
   */
  NDObject *Load(void *addr, IntArrayRef *shape, DataType type);

  /**
   * @brief Create a global memory access handle without emitting a load op.
   * @param addr memory address of tensor in global memory.
   * @param shape shape reference of tensor.
   * @param type data type of tensor.
   * @return the global access handle.
   */
  NDObject *GlobalAccess(void *addr, IntArrayRef *shape, DataType type);

  /**
   * @brief Emit a incontinuous load operation from input tensor.
   * @param addr memory address of input tensor. also can be relocated at codegen stage.
   * @param shape shape reference of input tensor.
   * @param stride stride reference of input tensor.
   * @param dtype data type of input tensor.
   * @return the result load operation.
   */
  NDObject *Load(void *addr, IntArrayRef *shape, IntArrayRef *stride, DataType type);

  /**
   * @brief Emit a gather load operation from input tensor and index tensor.
   * index should be created by GlobalAccess.
   * @param addr memory address of input tensor.
   * @param shape input tensor shape.
   * @param index global access object of index tensor.
   * @param type data type of input tensor.
   * @param gather_mode gather mode.
   * @return the result gather load operation.
   */
  NDObject *GatherLoad(void *addr, IntArrayRef *shape, NDObject *index, int axis, DataType type,
                       GatherMode gather_mode = kElementGather);

  /**
   * @brief Emit a slice load operation from input tensor. please use incontinuous load.
   */
  NDObject *SliceLoad(void *addr, IntArrayRef *shape, IntArrayRef *start, IntArrayRef *size, DataType type);

  /**
   * @brief [DEPRECATED] Emit a strided slice load operation from input tensor. please use incontinuous load.
   */
  NDObject *StridedSliceLoad(void *addr, IntArrayRef *shape, IntArrayRef *start, IntArrayRef *end, IntArrayRef *step, DataType type);

  /**
   * @brief [PRIVITE]
   */
  NDObject *MultiLoad(void *addr, IntArrayRef *shape, DataType type, const Comm *comm);

  /**
   * @brief Emit a store operation to output tensor.
   * @param addr memory address of output tensor. also can be relocated at codegen stage.
   * @param input source operation to store.
   * @return the result store operation.
   */
  NDObject *Store(void *addr, NDObject *input);

  /**
   * @brief Emit a incontinuous store operation to output tensor.
   * @param addr memory address of output tensor. also can be relocated at codegen stage.
   * @param input source operation to store.
   * @param stride stride reference of output tensor.
   * @return the result store operation.
   */
  NDObject *Store(void *addr, NDObject *input, IntArrayRef *stride);

  /**
   * @brief [PRIVITE]
   */
  NDObject *PadStore(void *addr, NDObject *input, int64_t pad_size);

  /**
   * @brief mark a store operation is an inplace store. used to prevent memory reuse for workspace.
   * @param store store operation.
   */
  void SetStoreInplace(NDObject *store);

  void SetStoreTemp(NDObject *store);
  void SetLoadBind(NDObject *load, NDObject *access);

  /**
   * @brief Emit a unary operation.
   * @param op_type unary operation type.
   * @param input operation input.
   * @return the result operation.
   */
  template <UnaryType op_type>
  NDObject *Unary(NDObject *input);

  /**
   * @brief Emit a binary operation.
   * @param op_type binary operation type.
   * @param lhs left head side input. supported data types: NDObject *, float, int32_t, Float16, BFloat16, ScalarRef *, int64_t(compare op only).
   * @param rhs right head side input. supported data types: NDObject *, float, int32_t, Float16, BFloat16, ScalarRef *, int64_t(compare op only).
   * @return the result operation.
   */
  template <BinaryType op_type, typename L, typename R>
  NDObject *Binary(L lhs, R rhs);

  /**
   * @brief Emit a reduce operation.
   * @param op_type reduce operation type.
   * @param input input operation to reduce.
   * @param dims reduce dims reference.
   * @param keepdims keep reduce dim in output shape.
   * @return the result operation.
   */
  template <ReduceType op_type>
  NDObject *Reduce(NDObject *input, IntArrayRef *dims, bool keepdims) { return _Reduce(op_type, input, dims, keepdims); }

  /**
   * @brief Emit a select operation.
   * @param cond condition operation.
   * @param lhs left head side input.
   * @param rhs right head side input.
   * @return the result operation.
   */
  NDObject *Select(NDObject *cond, NDObject *lhs, NDObject *rhs);

  /**
   * @brief Emit a cast operation.
   * @param input input operation to cast.
   * @param dtype target data type.
   * @return the result operation.
   */
  NDObject *Cast(NDObject *input, DataType type);

  /**
   * @brief Emit a broadcast operation.
   * @param input input operation.
   * @param shape target shape reference.
   * @return the result operation.
   */
  NDObject *Broadcast(NDObject *input, IntArrayRef *shape);

  /**
   * @brief Emit a scalar broadcast operation.
   * @param val broadcast value. support data type: float, int32_t, Float16, BFloat16, ScalarRef *.
   * @param shape broadcast shape reference.
   * @param dtype broadcast data type.
   * @return the result operation.
   */
  template <typename T>
  NDObject *Broadcast(T val, IntArrayRef *shape, DataType type);

  /**
   * @brief Emit a reshape operation. partially supported of kVector and kMix kernel type.
   * @param input input operation.
   * @param shape target shape reference.
   * @return the result operation.
   */
  NDObject *Reshape(NDObject *input, IntArrayRef *shape);

  /**
   * @brief Emit a permute operation. dims[i] means new dim i comes from old dim dims[i].
   * @param input input operation.
   * @param dims permutation axes reference (PyTorch convention).
   * @return the result operation.
   */
  NDObject *Permute(NDObject *input, IntArrayRef *dims);

  /**
   * @brief Emit a slice op. ONLY for spec vector.
   * @param input input operation.
   * @param start dimension offset to slice.
   * @param size dimension size to slice.
   * @return the result operation.
   */
  NDObject *Slice(NDObject *input, IntArrayRef *start, IntArrayRef *size);

  /**
   * @brief Emit a dim slice op. Slice along a single dimension.
   * @param input input operation.
   * @param dim dimension index to slice along.
   * @param begin start index (supports negative indexing).
   * @param end end index (supports negative indexing).
   * @return the result operation.
   */
  NDObject *Slice(NDObject *input, int dim, ScalarRef *begin, ScalarRef *end);

  /**
   * @brief Emit a copy operation.
   * @param input input operation.
   * @return the result operation.
   */
  NDObject *Copy(NDObject *input);

  /**
   * @brief Emit a concat operation. ONLY for view vector.
   * @param inputs array of input operations.
   * @param input_num number of inputs.
   * @param dim dimension to concatenate along.
   * @return the result operation.
   */
  NDObject *Concat(NDObject **inputs, size_t input_num, int dim);

  /**
   * @brief Emit a split operation. Split with equal-sized chunks along `dim` (the last chunk may be smaller).
   * @param input the input to split.
   * @param dim dimension to split along.
   * @param split_size size of each chunk along `dim`.
   * @param split_num number of chunk.
   * @return pointer to the output array.
   */
  NDObject **Split(NDObject *input, int dim, int64_t split_size, size_t split_num);

  /**
   * @brief Emit a OneHot operation.
   * @param indices input indices operation.
   * @param depth depth reference. the size should keep to 1.
   * @param axis position to insert the value.
   * @param on_value value to fill in output when indices on. supported data type: float, int32_t, Float16, BFloat16.
   * @param on_value value to fill in output when indices off. supported data type: float, int32_t, Float16, BFloat16.
   * @return the result operation.
   */
  template <typename T>
  NDObject *OneHot(NDObject *indices, IntArrayRef *depth, int axis, T on_value, T off_value);

  /**
   * @brief [PRIVITE]
   */
  NDObject *ElemAny(NDObject *input);

  /**
   * @brief Emit a cube matmul operation. support kFloat16 and kBFloat16.
   * @param lhs left input.
   * @param rhs right input.
   * @param trans_a transpose left input.
   * @param trans_b transpose right input.
   * @param bias bias input.
   * @return the result operation.
   */
  NDObject *MatMul(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b, NDObject *bias);

  /**
   * @brief Emit a cube grouped matmul operation. support kFloat16 and kBFloat16.
   * @param lhs left input.
   * @param rhs right input.
   * @param trans_a transpose left input.
   * @param trans_b transpose right input.
   * @param bias bias input.
   * @param group_list group list input.
   * @param group_type group split type.
   * @param group_list_type group list type.
   * @return the result operation.
   */
  NDObject *GroupedMatMul(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b, NDObject *bias,
                          NDObject *group_list, GmmSplitType group_type, GmmListType group_list_type);

  /**
   * @brief [PRIVITE] collective communication
   */
  template <ReduceType op_type>
  NDObject *AllReduce(NDObject *input, const Comm *comm) { return _AllReduce(op_type, input, comm); }
  NDObject *AllGather(NDObject *input, const Comm *comm);
  NDObject *AllGatherV2(NDObject *input, const Comm *comm);
  NDObject *ReduceScatter(NDObject *input, const Comm *comm);

  /**
   * @brief Get a external output from a multi-output op.
   * @param op the multi-output op.
   * @param index external output index.
   * @return the result operation for the specified output.
   */
  NDObject *ExtOut(NDObject *op, int index);

  /**
   * @brief switch to next speculate stage. support with kVector  with kSpeculate flag.
   */
  void SpecNext();

  /**
   * @brief add a new parallel sub kenrel. support with kParallel kernel type.
   * @param type sub kernel type. current only support kVector type.
   * @param flags sub kernel flags.
   * @param thread_limit sub kernel maximum thread limit. auto assign if 0.
   */
  void ParallelAdd(KernelType type, uint32_t flags, size_t thread_limit = 0);

  /**
   * @brief add a new sequence sub kenrel. support with kSequence kernel type.
   * @param type sub kernel type. supported types: kVector, kCube, kMix.
   * @param flags sub kernel flags.
   */
  void SequenceAdd(KernelType type, uint32_t flags);

  /**
   * @brief normalize and re-infer compute shapes of each operations. should called before codegen if any input shape changed.
   */
  void Normalize();

  /**
   * @brief generate code for current compute shapes. MUST called after Normalize.
   * @param relocs relocated entries for global tensor load/store operations with memory address changed.
   * @param reloc_size relocated entries size.
   * @param ws_alloc workspace allocator callback. use kUnifyWS kernel flags if has single workspace limit.
   */
  void CodeGen(const RelocEntry *relocs, size_t reloc_size, WsAllocator *ws_alloc);

  /**
   * @brief launch kernel to device to run. MUST called after CodeGen.
   * @param stream running stream.
   * @return 0 if success.
   */
  int Launch(void *stream);

  /**
   * @brief Clear kernel context. ONLY for kEager.
   */
  void Clear();

  /**
   * @brief Early codegen before workspace and input/output memory determined. kEager and kSplit is not supported!
   * @return workspace size.
   */
  size_t PreCodeGen();

  /**
   * @brief Launch for early codegen.
   * @param relocs relocated entries for global tensor load/store operations with memory address changed.
   * @param reloc_size relocated entries size.
   * @param workspace workspace memory address. should be allocated by return size of PreCodeGen.
   * @return 0 if success.
   */
  int Launch(const RelocEntry *relocs, size_t reloc_size, void *workspace, void *stream);

  /**
   * @brief Get operation shape. MUST called after Normalize or PreCodeGen.
   * @param op operation to get.
   * @return shape reference.
   */
  static IntArrayRef *GetShape(NDObject *op);

  /**
   * @brief Get operation dtype.
   * @param op operation to get.
   * @return result dtype.
   */
  static DataType GetDType(NDObject *op);

  /**
   * @brief Dump operation represent of this kernel.
   * @return result represent string. managed by DVM, DONOT delete.
   */
  const char *Dump() const;

  /**
   * @brief Dump operation represent of this kernel with ref shape.
   * @return result represent string. managed by DVM, DONOT delete.
   */
  const char *DumpR() const;

  /**
   * @brief Dump disassemblng string of this kernel. MUST called after CodeGen or PreCodeGen.
   * @return result disassemblng string. managed by DVM, DONOT delete.
   */
  const char *Das() const;

  /**
   * @brief Inspect current kernel and do validation check.
   * @return result of inspection report.
   */
  const char *Inspect() const;

  /**
   * @brief Get implement kernel.
   */
  VKernel *GetImpl() const { return kernel_; }

 protected:
  NDObject *_Reduce(int op_type, NDObject *input, IntArrayRef *dims, bool keepdims);
  NDObject *_AllReduce(int op_type, NDObject *input, const Comm *comm);

  VKernel *kernel_;

  /* DEPRECATED: for backward compatibility */
 public:
  NDObject *Unary(int op_type, NDObject *input) {
    switch (op_type) {
      case UnaryType::kSqrt:
        return Unary<UnaryType::kSqrt>(input);
      case UnaryType::kAbs:
        return Unary<UnaryType::kAbs>(input);
      case UnaryType::kLog:
        return Unary<UnaryType::kLog>(input);
      case UnaryType::kExp:
        return Unary<UnaryType::kExp>(input);
      case UnaryType::kReciprocal:
        return Unary<UnaryType::kReciprocal>(input);
      case UnaryType::kIsFinite:
        return Unary<UnaryType::kIsFinite>(input);
      case UnaryType::kLogicalNot:
        return Unary<UnaryType::kLogicalNot>(input);
      case UnaryType::kRound:
        return Unary<UnaryType::kRound>(input);
      case UnaryType::kFloor:
        return Unary<UnaryType::kFloor>(input);
      case UnaryType::kCeil:
        return Unary<UnaryType::kCeil>(input);
      case UnaryType::kTrunc:
        return Unary<UnaryType::kTrunc>(input);
      default:
        return nullptr;
    }
  }
  template <typename L, typename R>
  NDObject *Binary(int op_type, L lhs, R rhs) {
    switch (op_type) {
      case BinaryType::kEqual:
        return Binary<BinaryType::kEqual>(lhs, rhs);
      case BinaryType::kNotEqual:
        return Binary<BinaryType::kNotEqual>(lhs, rhs);
      case BinaryType::kGreater:
        return Binary<BinaryType::kGreater>(lhs, rhs);
      case BinaryType::kGreaterEqual:
        return Binary<BinaryType::kGreaterEqual>(lhs, rhs);
      case BinaryType::kLess:
        return Binary<BinaryType::kLess>(lhs, rhs);
      case BinaryType::kLessEqual:
        return Binary<BinaryType::kLessEqual>(lhs, rhs);
      case BinaryType::kAdd:
        return Binary<BinaryType::kAdd>(lhs, rhs);
      case BinaryType::kSub:
        return Binary<BinaryType::kSub>(lhs, rhs);
      case BinaryType::kMul:
        return Binary<BinaryType::kMul>(lhs, rhs);
      case BinaryType::kDiv:
        return Binary<BinaryType::kDiv>(lhs, rhs);
      case BinaryType::kPow:
        return Binary<BinaryType::kPow>(lhs, rhs);
      case BinaryType::kMaximum:
        return Binary<BinaryType::kMaximum>(lhs, rhs);
      case BinaryType::kMinimum:
        return Binary<BinaryType::kMinimum>(lhs, rhs);
      case BinaryType::kLogicalAnd:
        return Binary<BinaryType::kLogicalAnd>(lhs, rhs);
      case BinaryType::kLogicalOr:
        return Binary<BinaryType::kLogicalOr>(lhs, rhs);
      default:
        return nullptr;
    }
  }
  NDObject *Reduce(int op_type, NDObject *input, IntArrayRef *dims, bool keepdims) { return _Reduce(op_type, input, dims, keepdims); }
  NDObject *AllReduce(int op_type, NDObject *input, const Comm *comm) { return _AllReduce(op_type, input, comm); }
  void ParallelNext() { return ParallelAdd(KernelType::kVector, 0); }
  void Infer() { Normalize(); }
  size_t CodeGen() { return PreCodeGen(); }
};

/**
 * @brief DVM Global configure API.
 */
class DVM_API Config {
 public:
  Config() = default;
  ~Config() = default;
  static Config &Instance();
  bool IsInitialized() const;
  virtual Config &SetDeterm() = 0;
  virtual Config &UnsetDeterm() = 0;
  virtual Config &SetOnlineTuner() = 0;
  virtual Config &UnsetOnlineTuner() = 0;
  virtual Config &SetLazyTuner() = 0;
  virtual Config &UnsetLazyTuner() = 0;
  virtual Config &SetVfFusion(int mode = 1) = 0;
  virtual Config &UnsetVfFusion() = 0;
};

/* DEPRECATED: for backward compatibility */
using DType = DataType;
using UnaryOpType = UnaryType;
using BinaryOpType = BinaryType;
using ReduceOpType = ReduceType;
using GroupType = GmmSplitType;
using GroupListType = GmmListType;
using ShapeRef = IntArrayRef;

[[noreturn]] DVM_API void DvmException(const char *error_str);
}  // namespace dvm
#endif  // _DVM_H_
