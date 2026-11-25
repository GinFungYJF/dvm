VPATH = ./src:./include

DVM_ARCH ?= npu
DVM_PY_LINK ?= static
PYTHON ?= python3
PYBIND11_INCLUDES ?= $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null)
.DEFAULT_GOAL := all
ifneq ($(filter $(DVM_ARCH),npu gpu),$(DVM_ARCH))
$(error DVM_ARCH must be npu or gpu)
endif
ifneq ($(filter $(DVM_PY_LINK),static shared),$(DVM_PY_LINK))
$(error DVM_PY_LINK must be static or shared)
endif

OBJ = ops.o ops_m.o ops_c.o kernel.o xkernel.o gkernel.o code.o dvm.o schedule.o pass.o utils.o vf_fusion.o system.o

ifeq ($(DVM_ARCH),gpu)
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= sm_70
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_TARGET_DIR := $(CUDA_HOME)/targets/$(shell uname -m)-linux
CUDA_INCLUDE_DIR ?= $(if $(wildcard $(CUDA_HOME)/include/cuda_runtime.h),$(CUDA_HOME)/include,$(CUDA_TARGET_DIR)/include)
CUDA_LIB_DIR ?= $(if $(wildcard $(CUDA_HOME)/lib64/libcudart.so),$(CUDA_HOME)/lib64,$(CUDA_HOME)/lib)
CFLAGS = --std=c++17 -Werror -Wall -I./include -I$(CUDA_INCLUDE_DIR) ${DVM_CUSTOM_FLAGS} -fPIC -fvisibility=hidden -DDVM_ARCH_GPU
CFLAGS += -fstack-protector-all -Wno-array-bounds
LDFLAGS += -Wl,-z,relro,-z,now,-z,noexecstack
LD_FLAGS = -L$(CUDA_LIB_DIR) -lcudart
else
OBJ += msprof.o comm.o
CFLAGS = --std=c++17 -Werror -Wall -I./include -I${ASCEND_HOME_PATH}/include -I${ASCEND_HOME_PATH}/pkg_inc ${DVM_CUSTOM_FLAGS} -fPIC -fvisibility=hidden
CFLAGS += -fstack-protector-all -Wno-array-bounds
LDFLAGS += -Wl,-z,relro,-z,now,-z,noexecstack
CCE_FLGAS_C220 = --std=c++17 -Wno-int-to-pointer-cast\
				 --cce-aicore-only\
                 --cce-auto-sync=off \
                 -mllvm -cce-aicore-function-stack-size=16000 \
                 -mllvm -cce-aicore-record-overflow=false \
                 -mllvm -cce-aicore-addr-transform \
                 -mllvm -cce-aicore-jump-expand=true \
                 -mllvm -cce-aicore-mask-opt=false

CCE_FLGAS_C310 = --std=c++17 -Wno-int-to-pointer-cast\
                 --cce-aicore-only \
                 --cce-auto-sync=off \
                 --cce-simd-vf-fusion=true \
                 -mllvm -cce-aicore-stack-size=0x8000 \
                 -mllvm -cce-aicore-function-stack-size=0x8000 \
                 -mllvm -cce-aicore-addr-transform \
                 -mllvm -cce-aicore-or-combine=false \
                 -mllvm -instcombine-code-sinking=false \
                 -mllvm -cce-aicore-jump-expand=true \
                 -mllvm -cce-aicore-mask-opt=false \
                 -mllvm -cce-aicore-dcci-insert-for-scalar=false

ASCENDC_FLAGS = -I${ASCEND_HOME_PATH}/asc -I${ASCEND_HOME_PATH}/asc/include \
                -I${ASCEND_HOME_PATH}/asc/include/basic_api -I${ASCEND_HOME_PATH}/asc/impl/basic_api

ifneq ($(DVM_SOC_SIMU),)
LD_FLAGS = -L${ASCEND_HOME_PATH}/tools/simulator/${DVM_SOC_SIMU}/lib -lruntime_camodel -L${ASCEND_HOME_PATH}/lib64 -lascendcl
CFLAGS += -DVK_SIM_MODEL
else
LD_FLAGS = -L${ASCEND_HOME_PATH}/lib64 -lascendcl
endif
LD_FLAGS += -lprofapi -ldl
endif

CXX_VERSION := $(shell $(CXX) --version 2>/dev/null | head -n 1)
ifneq (,$(findstring clang,$(CXX_VERSION)))
CFLAGS += -Wno-error
endif

ifneq ($(dbg),)
CFLAGS += -g -O0 -DDEBUG
CCE_FLGAS_C220 += -DDEBUG
CCE_FLGAS_C310 += -DDEBUG
else
CFLAGS += -O2 -D_FORTIFY_SOURCE=2
endif

ifneq ($(asan),)
CFLAGS += -fsanitize=address -fsanitize-recover=address -fno-omit-frame-pointer
LDFLAGS += -fsanitize=address
endif

XXD_FOUND := $(shell which xxd 2>/dev/null)
ifdef XXD_FOUND
XXDI = xxd -i
else
XXDI = python3 scripts/xxdi.py
endif

HEADERS = $(OBJ:.o=.h) isa.h runtime.h dvm_export.h
ifeq ($(DVM_ARCH),gpu)
HEADERS += vm_gpu.h
endif

.PHONY: all libs shared clean help FORCE check-toolchain check-shared install
all: libs _dvm_py.so
libs: libdvm.a libdvm.so
shared: libdvm.so
check-shared: all
	$(PYTHON) scripts/check_shared.py

# Record build settings so changing backend, compiler or linkage cannot reuse stale objects.
.dvm-build-config: FORCE
	@$(PYTHON) scripts/build_config.py $@ '$(DVM_ARCH)' '$(CXX)' '$(CFLAGS)' '$(NVCC)' '$(CUDA_ARCH)' '$(PRE_ASCEND)' '$(DVM_PY_LINK)' '$(PYBIND11_INCLUDES)' '$(LDFLAGS)' '$(LD_FLAGS)'

$(OBJ) vm.o pybind_api.o dry_run.o: .dvm-build-config

ifeq ($(DVM_PY_LINK),shared)
_dvm_py.so: pybind_api.o dry_run.o libdvm.so
	$(CXX) -shared $(LDFLAGS) pybind_api.o dry_run.o -L. -ldvm $(LD_FLAGS) -Wl,-rpath,'$$ORIGIN' -o $@
	cp $@ libdvm.so.1 ./python/dvm/
else
_dvm_py.so: pybind_api.o dry_run.o libdvm.a
	$(CXX) -shared $(LDFLAGS) $^ $(LD_FLAGS) -o $@
	cp $@ ./python/dvm/
endif

libdvm.a: $(OBJ) vm.o
	rm -f $@
	$(AR) crs $@ $^

libdvm.so.1: libdvm.a scripts/dvm.map
	$(CXX) -shared $(LDFLAGS) -Wl,-z,defs -Wl,-soname,libdvm.so.1 -Wl,--version-script=scripts/dvm.map -Wl,--whole-archive libdvm.a -Wl,--no-whole-archive $(LD_FLAGS) -o $@

libdvm.so: libdvm.so.1
	ln -sf libdvm.so.1 $@

PREFIX ?= /usr/local
install: libs
	install -d $(DESTDIR)$(PREFIX)/lib $(DESTDIR)$(PREFIX)/include
	install -m 644 libdvm.a libdvm.so.1 $(DESTDIR)$(PREFIX)/lib/
	ln -sf libdvm.so.1 $(DESTDIR)$(PREFIX)/lib/libdvm.so
	install -m 644 include/dvm.h include/dvm_export.h $(DESTDIR)$(PREFIX)/include/

pybind_api.o: pybind_api.cc pybind_api.h dvm_py.h $(HEADERS)
	$(CXX) -c $(CFLAGS) $(PYBIND11_INCLUDES) $< -o $@

dry_run.o: dry_run.cc isa.h vm_aiv.cce vm_aic.cce $(HEADERS)
	$(CXX) -c $(CFLAGS) $< -o $@

${OBJ}: %.o: %.cc $(HEADERS)
	$(CXX) -c $(CFLAGS) $< -o $@

ifeq ($(DVM_ARCH),gpu)
$(OBJ) vm.o pybind_api.o dry_run.o: | check-toolchain

check-toolchain:
	$(PYTHON) scripts/check_cuda.py "$(NVCC)"

vm.o: vm_gpu.cu isa.h vm_gpu.h $(GPU_RUNOP_CHECK) | check-toolchain
	python3 scripts/check_gpu_dispatch.py src/vm_gpu.h src/vm_gpu.cu
	$(NVCC) -c -O2 --std=c++17 -DDVM_ARCH_GPU -arch=$(CUDA_ARCH) --jump-table-density=60 -Xcompiler -fPIC -I./include -I./src $< -o $@
else
VMAIN_OFFSET=0x$$(llvm-objdump -t vm_aic_c220.o | grep " dvm_mix_aic$$" | awk '{print $$5}')
VMAIN_C310_OFFSET=0x$$(llvm-objdump -t vm_aic_c310.o | grep " dvm_mix_aic$$" | awk '{print $$5}')

vm.o: g_vkernel_c220_bin g_vkernel_c310_bin
	echo "extern const" > vm.cc
	$(XXDI) g_vkernel_c220_bin >> vm.cc
	$(XXDI) g_vkernel_c310_bin >> vm.cc
	objdump -t g_vkernel_c310_bin | grep " F " | python3 scripts/find_addrs.py src/isa.h c310 >> vm.cc
	objdump -t g_vkernel_c220_bin | grep " F " | python3 scripts/find_addrs.py src/isa.h c220 >> vm.cc
	python3 scripts/find_meta.py g_vkernel_c310_bin >> vm.cc
	$(CXX) -c $(CFLAGS) vm.cc -o vm.o

ifneq ($(PRE_ASCEND),)
g_vkernel_c220_bin: prebuild/g_vkernel_c220_bin
	cp -f $< $@
g_vkernel_c310_bin: prebuild/g_vkernel_c310_bin
	cp -f $< $@
else
g_vkernel_c220_bin: vm_aiv_c220.o vm_aic_c220.o
	ld.lld -Ttext=0 vm_aic_c220.o vm_aiv_c220.o -static -o g_vkernel_c220_bin
g_vkernel_c310_bin: vm_aiv_c310.o vm_aic_c310.o
	ld.lld -Ttext=0 vm_aic_c310.o vm_aiv_c310.o -static -o g_vkernel_c310_bin
endif

vm_aiv_c220.o: vm_aiv.cce isa.h vm_aiv.h vm_aic_c220.o
	ccec -c -O2 $(CCE_FLGAS_C220) -D VMAIN_OFFSET=$(VMAIN_OFFSET) --cce-aicore-arch=dav-c220-vec src/vm_aiv.cce -o vm_aiv_c220.o

vm_aic_c220.o: vm_aic.cce isa.h vm_aic.h
	ccec -c -O2 $(CCE_FLGAS_C220) --cce-aicore-arch=dav-c220-cube src/vm_aic.cce -o vm_aic_c220.o

vm_aiv_c310.o: vm_aiv_c310.cce isa.h vm_aiv.h vm_aic_c310.o
	ccec -c -O2 $(CCE_FLGAS_C310) $(ASCENDC_FLAGS) -D VMAIN_OFFSET=$(VMAIN_C310_OFFSET) --cce-aicore-arch=dav-c310-vec src/vm_aiv_c310.cce -o vm_aiv_c310.o

vm_aic_c310.o: vm_aic_c310.cce isa.h vm_aic.h
	ccec -c -O2 $(CCE_FLGAS_C310) --cce-aicore-arch=dav-c310-cube src/vm_aic_c310.cce -o vm_aic_c310.o

prebuild: g_vkernel_c220_bin g_vkernel_c310_bin
	cp -f g_vkernel_c220_bin prebuild/g_vkernel_c220_bin
	cp -f g_vkernel_c310_bin prebuild/g_vkernel_c310_bin
	printf '[lib information]\ngit branch: %s\ncommit  id: %s\n' "$$(git branch --show-current)" "$$(git rev-parse HEAD)" > prebuild/lib_info.txt
endif

clean:
	rm -f *.o *.so *.a *bin vm.cc

help:
	@echo "Usage: make [dbg=1] [asan=1] [PRE_ASCEND=1]"
