# DNNF90 is a library first (lib/), with the PINN trainer as one bundled
# application (app/).  host/ holds host-side reference code (descriptors).
#
# All build products (objects, modules, executables, archives) go to
# build/.  The usual names still work as phony aliases, so
#   make            builds build/serial.out
#   make mpi        builds build/mpi.out
#   make lib        builds build/libdnnf90.a
#   make train_example.out   builds build/train_example.out
# and so on.  make clean removes build/ entirely.
VPATH = lib:app:host:tools

# The library must be reentrant regardless of how the host threads:
# -frecursive guarantees stack allocation of all locals, so one work
# space per thread is the only requirement for concurrent calls.
# Optional BLAS path for the library kernels: make ... BLAS=1 compiles the
# three width-quadratic contractions as dgemm (5-7x at force-field widths)
# and links -lopenblas.  Off by default: the loop kernels are the bitwise
# reference; the BLAS path agrees to roundoff.
BLAS ?= 0
ifeq (${BLAS},1)
BLASDEF = -DUSE_BLAS
BLASLIB = -lopenblas
else
BLASDEF =
BLASLIB =
endif

B = build
F90 = gfortran -O3 -g -fbacktrace
FFLAGS = -cpp
MODFLAGS = -J${B} -I${B}
LIBFLAGS = ${FFLAGS} ${BLASDEF} -frecursive

APP_OBJ = ${B}/parallel_module.o ${B}/global_variables.o ${B}/rand_module.o \
	${B}/blas_wrap_module.o ${B}/bessel_module.o ${B}/erf_value_module.o ${B}/multi_index_bell_module.o ${B}/net_module.o ${B}/train_module.o \
	${B}/kalman_module.o ${B}/committee_module.o ${B}/lib_net_module.o ${B}/committee_run_module.o ${B}/pinn_module.o \
	${B}/io_module.o ${B}/init_weight_module.o ${B}/optimizer_module.o \
	${B}/validation_module.o ${B}/hod_check_module.o ${B}/sgd_batch_module.o

COMMON_MOD = app/parallel_module.f90 app/global_variables.f90 app/rand_module.f90 \
	lib/blas_wrap_module.f90 lib/bessel_module.f90 lib/erf_value_module.f90 lib/multi_index_bell_module.f90 lib/net_module.f90 lib/train_module.f90 \
	lib/kalman_module.f90 lib/committee_module.f90 app/lib_net_module.f90 app/committee_run_module.f90 app/pinn_module.f90 \
	app/io_module.f90 app/init_weight_module.f90 app/optimizer_module.f90 \
	app/validation_module.f90 app/hod_check_module.f90 app/sgd_batch_module.f90 app/main.f90

LIB_OBJ = ${B}/blas_wrap_module.o ${B}/bessel_module.o ${B}/erf_value_module.o ${B}/multi_index_bell_module.o ${B}/net_module.o ${B}/train_module.o \
	${B}/committee_module.o ${B}/kalman_module.o

default : ${B}/serial.out

${B} :
	mkdir -p ${B}

${B}/%.o : %.f90 | ${B}
	${F90} -c $< ${FFLAGS} ${BLASDEF} ${MODFLAGS} -o $@

# module dependencies
# Every object lists the objects whose .mod files it reads, derived from
# the USE statements, so that a parallel make builds them in an order
# that works.  The COMMON_MOD line alone only says "rebuild when any
# source changes"; it does not order the compilations, which is why
# `make -j` used to stop on a missing .mod.
${APP_OBJ} ${B}/main.o : ${COMMON_MOD}
${B}/multi_index_bell_module.o : ${B}/bessel_module.o ${B}/erf_value_module.o
${B}/net_module.o        : ${B}/blas_wrap_module.o ${B}/multi_index_bell_module.o
${B}/train_module.o      : ${B}/blas_wrap_module.o ${B}/multi_index_bell_module.o ${B}/net_module.o
${B}/committee_module.o  : ${B}/multi_index_bell_module.o ${B}/net_module.o
${B}/kalman_module.o     : ${B}/net_module.o ${B}/train_module.o
${B}/c_api_module.o      : ${B}/multi_index_bell_module.o ${B}/net_module.o ${B}/train_module.o
${B}/rand_module.o       : ${B}/global_variables.o ${B}/parallel_module.o
${B}/pinn_module.o       : ${B}/global_variables.o ${B}/multi_index_bell_module.o
${B}/lib_net_module.o    : ${B}/global_variables.o ${B}/kalman_module.o ${B}/multi_index_bell_module.o ${B}/net_module.o ${B}/train_module.o
${B}/committee_run_module.o : ${B}/global_variables.o ${B}/committee_module.o ${B}/multi_index_bell_module.o ${B}/net_module.o
${B}/io_module.o         : ${B}/global_variables.o ${B}/lib_net_module.o ${B}/parallel_module.o ${B}/rand_module.o ${B}/multi_index_bell_module.o
${B}/init_weight_module.o : ${B}/global_variables.o ${B}/io_module.o ${B}/parallel_module.o ${B}/rand_module.o
${B}/optimizer_module.o  : ${B}/global_variables.o ${B}/parallel_module.o
${B}/validation_module.o : ${B}/global_variables.o ${B}/io_module.o ${B}/lib_net_module.o ${B}/parallel_module.o ${B}/pinn_module.o ${B}/multi_index_bell_module.o
${B}/hod_check_module.o  : ${B}/global_variables.o ${B}/lib_net_module.o ${B}/pinn_module.o ${B}/multi_index_bell_module.o
${B}/sgd_batch_module.o  : ${B}/global_variables.o ${B}/init_weight_module.o ${B}/io_module.o ${B}/lib_net_module.o ${B}/optimizer_module.o ${B}/parallel_module.o ${B}/pinn_module.o ${B}/rand_module.o ${B}/validation_module.o ${B}/blas_wrap_module.o ${B}/multi_index_bell_module.o
${B}/main.o              : ${B}/committee_run_module.o ${B}/global_variables.o ${B}/hod_check_module.o ${B}/io_module.o ${B}/parallel_module.o ${B}/sgd_batch_module.o
${B}/api_module.o        : ${APP_OBJ}
${B}/symfunc_module.o    : | ${B}

${B}/serial.out : ${APP_OBJ} ${B}/main.o
	${F90} -o $@ ${APP_OBJ} ${B}/main.o ${BLASLIB}
serial.out : ${B}/serial.out

# embeddable library
lib : ${B}/libdnnf90.a
${B}/libdnnf90.a : ${LIB_OBJ} ${B}/c_api_module.o
	ar rcs $@ ${LIB_OBJ} ${B}/c_api_module.o
shared : ${LIB_OBJ} ${B}/c_api_module.o
	${F90} -shared -fPIC ${LIBFLAGS} ${MODFLAGS} -o ${B}/libdnnf90.so \
	  lib/bessel_module.f90 lib/erf_value_module.f90 lib/multi_index_bell_module.f90 lib/net_module.f90 lib/train_module.f90 \
	  lib/committee_module.f90 lib/kalman_module.f90 lib/c_api_module.f90 ${BLASLIB}

# examples (each also available under its bare name)
${B}/embed_example.out : ${APP_OBJ} ${B}/api_module.o tools/example_embed.f90
	${F90} ${FFLAGS} ${MODFLAGS} -o $@ ${APP_OBJ} ${B}/api_module.o tools/example_embed.f90 ${BLASLIB}
embed_example.out : ${B}/embed_example.out

${B}/mlff_example.out : ${APP_OBJ} ${B}/api_module.o tools/example_mlff.f90
	${F90} ${FFLAGS} ${MODFLAGS} -fopenmp -o $@ ${APP_OBJ} ${B}/api_module.o tools/example_mlff.f90 ${BLASLIB}
mlff_example.out : ${B}/mlff_example.out

${B}/fdcheck.out : ${LIB_OBJ} tools/dbg_backward_fd.f90
	${F90} ${LIBFLAGS} ${MODFLAGS} -o $@ ${LIB_OBJ} tools/dbg_backward_fd.f90 ${BLASLIB}

# The GEMM/Bell split measurement.  Compiled with -DPHASE_TIMING, which
# the default library objects do not carry, so the library sources are
# rebuilt into this one executable with the flag; the shipped objects
# and every other build stay untouched.
${B}/fwd_grad_timing.out : lib/blas_wrap_module.f90 lib/bessel_module.f90 lib/erf_value_module.f90 lib/multi_index_bell_module.f90 lib/net_module.f90 lib/train_module.f90 tools/fwd_grad_timing.f90 | ${B}
	mkdir -p ${B}/ptmod
	${F90} ${LIBFLAGS} -DPHASE_TIMING -J${B}/ptmod -I${B}/ptmod -o $@ \
	  lib/blas_wrap_module.f90 lib/bessel_module.f90 lib/erf_value_module.f90 \
	  lib/multi_index_bell_module.f90 lib/net_module.f90 lib/train_module.f90 \
	  tools/fwd_grad_timing.f90 ${BLASLIB}
fwd_grad_timing.out : ${B}/fwd_grad_timing.out

# The directional counterpart of the framework comparison: the same
# loss bench/cpp/adolc_grad_bench.cpp differentiates (Taylor
# coefficients along x_1, orders 0..K), timed on the library kernels.
${B}/dir_grad_timing.out : ${LIB_OBJ} tools/dir_grad_timing.f90
	${F90} ${LIBFLAGS} ${MODFLAGS} -o $@ ${LIB_OBJ} tools/dir_grad_timing.f90 ${BLASLIB}
dir_grad_timing.out : ${B}/dir_grad_timing.out

# Every carried derivative of a trained network, at given points: the
# file a collocation run does not write (see the tool header).
${B}/hod_dump.out : ${LIB_OBJ} tools/hod_dump.f90
	${F90} ${LIBFLAGS} ${MODFLAGS} -o $@ ${LIB_OBJ} tools/hod_dump.f90 ${BLASLIB}
hod_dump.out : ${B}/hod_dump.out

# Forward pass only, the counterpart of fwd_grad_timing used for the
# ADOL-C tensor_eval comparison.
${B}/fwd_only_timing.out : ${LIB_OBJ} tools/fwd_only_timing.f90
	${F90} ${LIBFLAGS} ${MODFLAGS} -o $@ ${LIB_OBJ} tools/fwd_only_timing.f90 ${BLASLIB}
fwd_only_timing.out : ${B}/fwd_only_timing.out

# List-vs-padded benchmark of the Bell composition (see the tool header).
${B}/bell_pad_timing.out : ${LIB_OBJ} tools/bell_pad_timing.f90
	${F90} ${LIBFLAGS} ${MODFLAGS} -o $@ ${LIB_OBJ} tools/bell_pad_timing.f90 ${BLASLIB}
bell_pad_timing.out : ${B}/bell_pad_timing.out
fdcheck.out : ${B}/fdcheck.out

${B}/train_example.out : ${APP_OBJ} ${B}/api_module.o tools/example_train.f90
	${F90} ${FFLAGS} ${MODFLAGS} -fopenmp -o $@ ${APP_OBJ} ${B}/api_module.o tools/example_train.f90 ${BLASLIB}
train_example.out : ${B}/train_example.out

${B}/libverify_example.out : ${LIB_OBJ} tools/example_libverify.f90
	${F90} ${FFLAGS} ${BLASDEF} ${MODFLAGS} -frecursive -fopenmp -o $@ ${LIB_OBJ} tools/example_libverify.f90 ${BLASLIB}

${B}/taylorgreen_example.out : ${LIB_OBJ} ${B}/global_variables.o ${B}/pinn_module.o tools/example_taylorgreen.f90
	${F90} ${FFLAGS} ${BLASDEF} ${MODFLAGS} -frecursive -fopenmp -o $@ ${LIB_OBJ} ${B}/global_variables.o ${B}/pinn_module.o tools/example_taylorgreen.f90 ${BLASLIB}
libverify_example.out : ${B}/libverify_example.out

# Standalone correctness checks that live in tools/ and need only the
# library objects.
${B}/multiout_example.out : ${LIB_OBJ} tools/example_multiout.f90
	${F90} ${LIBFLAGS} ${MODFLAGS} -o $@ ${LIB_OBJ} tools/example_multiout.f90 ${BLASLIB}
multiout_example.out : ${B}/multiout_example.out

${B}/multiout_adj_example.out : ${LIB_OBJ} tools/example_multiout_adj.f90
	${F90} ${LIBFLAGS} ${MODFLAGS} -o $@ ${LIB_OBJ} tools/example_multiout_adj.f90 ${BLASLIB}
multiout_adj_example.out : ${B}/multiout_adj_example.out

${B}/act_roundtrip_example.out : ${LIB_OBJ} tools/example_act_roundtrip.f90
	${F90} ${LIBFLAGS} ${MODFLAGS} -o $@ ${LIB_OBJ} tools/example_act_roundtrip.f90 ${BLASLIB}
act_roundtrip_example.out : ${B}/act_roundtrip_example.out

${B}/lifecycle_switch_example.out : ${APP_OBJ} ${B}/api_module.o tools/example_lifecycle_switch.f90
	${F90} ${LIBFLAGS} ${MODFLAGS} -o $@ ${APP_OBJ} ${B}/api_module.o tools/example_lifecycle_switch.f90 ${BLASLIB}
lifecycle_switch_example.out : ${B}/lifecycle_switch_example.out

${B}/api_lifecycle_example.out : ${APP_OBJ} ${B}/api_module.o tools/example_api_lifecycle.f90
	${F90} ${LIBFLAGS} ${MODFLAGS} -o $@ ${APP_OBJ} ${B}/api_module.o tools/example_api_lifecycle.f90 ${BLASLIB}
api_lifecycle_example.out : ${B}/api_lifecycle_example.out

${B}/batch_act_example.out : ${LIB_OBJ} tools/example_batch_act.f90
	${F90} ${LIBFLAGS} ${MODFLAGS} -o $@ ${LIB_OBJ} tools/example_batch_act.f90 ${BLASLIB}
batch_act_example.out : ${B}/batch_act_example.out ${B}/act_roundtrip_example.out

${B}/product_adj_example.out : ${LIB_OBJ} ${B}/global_variables.o ${B}/pinn_module.o tools/example_product_adj.f90
	${F90} ${LIBFLAGS} ${MODFLAGS} -o $@ ${LIB_OBJ} ${B}/global_variables.o ${B}/pinn_module.o tools/example_product_adj.f90 ${BLASLIB}
product_adj_example.out : ${B}/product_adj_example.out

${B}/lbfgs_check_example.out : ${LIB_OBJ} ${B}/global_variables.o ${B}/optimizer_module.o tools/example_lbfgs_check.f90
	${F90} ${LIBFLAGS} ${MODFLAGS} -o $@ ${LIB_OBJ} ${B}/global_variables.o ${B}/optimizer_module.o tools/example_lbfgs_check.f90 ${BLASLIB}
lbfgs_check_example.out : ${B}/lbfgs_check_example.out

${B}/uq_example.out : ${LIB_OBJ} tools/example_uq.f90
	${F90} ${FFLAGS} ${BLASDEF} ${MODFLAGS} -frecursive -o $@ ${LIB_OBJ} tools/example_uq.f90 ${BLASLIB}
uq_example.out : ${B}/uq_example.out

${B}/customloss_example.out : ${B}/libdnnf90.a tools/example_customloss.f90
	${F90} ${FFLAGS} ${BLASDEF} ${MODFLAGS} -frecursive -o $@ tools/example_customloss.f90 ${B}/libdnnf90.a ${BLASLIB}
customloss_example.out : ${B}/customloss_example.out

${B}/hod_ff_example.out : ${LIB_OBJ} ${B}/symfunc_module.o tools/example_hod_ff.f90
	${F90} ${FFLAGS} ${BLASDEF} ${MODFLAGS} -frecursive -o $@ ${LIB_OBJ} ${B}/symfunc_module.o tools/example_hod_ff.f90 ${BLASLIB}
hod_ff_example.out : ${B}/hod_ff_example.out

${B}/roofline.out : tools/bench_roofline.f90 | ${B}
	${F90} ${FFLAGS} ${MODFLAGS} -o $@ tools/bench_roofline.f90 -lopenblas
roofline.out : ${B}/roofline.out

${B}/gen_c_ref.out : ${B}/bessel_module.o ${B}/erf_value_module.o ${B}/multi_index_bell_module.o ${B}/net_module.o tools/gen_c_ref.f90
	${F90} ${FFLAGS} ${MODFLAGS} -o $@ ${B}/bessel_module.o ${B}/erf_value_module.o ${B}/multi_index_bell_module.o ${B}/net_module.o tools/gen_c_ref.f90
gen_c_ref.out : ${B}/gen_c_ref.out

${B}/gen_hod.out : ${B}/bessel_module.o ${B}/erf_value_module.o ${B}/multi_index_bell_module.o tools/gen_hod_train.f90
	${F90} ${FFLAGS} ${MODFLAGS} -o $@ ${B}/bessel_module.o ${B}/erf_value_module.o ${B}/multi_index_bell_module.o tools/gen_hod_train.f90
gen_hod.out : ${B}/gen_hod.out

${B}/gen_pinn.out : tools/gen_pinn_data.f90 | ${B}
	${F90} ${FFLAGS} ${MODFLAGS} -o $@ tools/gen_pinn_data.f90
gen_pinn.out : ${B}/gen_pinn.out

# C interface examples
CC = gcc -O2
${B}/c_handles.out : ${B}/libdnnf90.a tools/example_c_handles.c lib/dnnf90.h
	cc -I lib -o $@ tools/example_c_handles.c ${B}/libdnnf90.a ${BLASLIB} -lgfortran -lm
c_handles.out : ${B}/c_handles.out

${B}/c_example.out : ${B}/libdnnf90.a tools/example_c.c lib/dnnf90.h
	${CC} -Ilib -o $@ tools/example_c.c ${B}/libdnnf90.a -lgfortran -lm ${BLASLIB}
c_example.out : ${B}/c_example.out

${B}/c_train_example.out : ${B}/libdnnf90.a tools/example_c_train.c lib/dnnf90.h
	${CC} -Ilib -o $@ tools/example_c_train.c ${B}/libdnnf90.a -lgfortran -lm ${BLASLIB}
c_train_example.out : ${B}/c_train_example.out

# MPI build (local SGD with periodic weight averaging): make mpi
MPIF90 = mpif90 -O2 -g -fbacktrace -fallow-argument-mismatch
mpi : | ${B}
	$(MPIF90) -cpp -D_MPI_ -J${B} -o ${B}/mpi.out $(COMMON_MOD)

# strict Fortran 2003 conformance check of every source file.
# The library was strict Fortran 95 until the derived-type array
# components were changed from POINTER to ALLOCATABLE: allocatable
# components are Fortran 2003 (TR 15581).  The change is worth it --
# it removes the compiler's aliasing assumptions on the work arrays,
# about 30 per cent of one gradient at -O3, and it makes leaks hard,
# since allocatable components are released automatically.  Nothing
# beyond TR 15581 and iso_c_binding is used, so a compiler with the
# allocatable-components TR accepts the whole library.  The former
# exceptions are no longer exceptions: c_api_module's iso_c_binding
# and the command-argument intrinsics of tools/gen_* are Fortran 2003.
# Instrumented build: bounds checking, uninitialized values poisoned with
# a signalling NaN, and IEEE traps.  This is the net for the defect class
# a finite-difference comparison cannot see (writes past an array, reads
# of never-set variables, silent division by zero).
harden :
	$(MAKE) clean
	$(MAKE) FFLAGS="-cpp -O0 -g -fbacktrace -fcheck=all -finit-real=snan \
	  -finit-integer=-99999999 -ffpe-trap=invalid,zero,overflow -frecursive"
	@echo "hardened build in ${B}/serial.out; run the cases you care about"

negtests : ${B}/serial.out ${B}/hod_ff_example.out ${B}/fdcheck.out ${B}/libverify_example.out ${B}/taylorgreen_example.out ${B}/product_adj_example.out ${B}/multiout_adj_example.out ${B}/batch_act_example.out ${B}/act_roundtrip_example.out
	@tools/negtests.sh

f2003check :
	@rm -rf ${B}/.f2003 && mkdir -p ${B}/.f2003
	@${F90} -cpp -frecursive -fsyntax-only -J${B}/.f2003 lib/bessel_module.f90 lib/erf_value_module.f90
	@for f in lib/blas_wrap_module.f90 lib/multi_index_bell_module.f90 lib/net_module.f90 \
	          lib/train_module.f90 lib/committee_module.f90 \
	          lib/kalman_module.f90; do \
	   ${F90} -std=f2003 -cpp -frecursive -fsyntax-only -J${B}/.f2003 $$f || exit 1; done
	@for f in app/parallel_module.f90 app/global_variables.f90 \
	          app/rand_module.f90 app/lib_net_module.f90 app/pinn_module.f90 \
	          app/io_module.f90 app/init_weight_module.f90 \
	          app/optimizer_module.f90 \
	          lib/committee_module.f90 app/committee_run_module.f90 \
	          app/hod_check_module.f90 app/validation_module.f90 \
	          app/sgd_batch_module.f90 app/api_module.f90 app/main.f90; do \
	   [ -f $$f ] && { ${F90} -std=f2003 -cpp -frecursive -fsyntax-only -I${B}/.f2003 -J${B}/.f2003 $$f || exit 1; }; done
	@for f in $$(ls host/*.f90); do \
	   ${F90} -std=f2003 -cpp -frecursive -fsyntax-only -I${B}/.f2003 -J${B}/.f2003 $$f || exit 1; done
	@for f in $$(ls tools/*.f90 | grep -v "gen_"); do \
	   ${F90} -std=f2003 -cpp -frecursive -fopenmp -fsyntax-only -I${B}/.f2003 -J${B}/.f2003 $$f || exit 1; done
	@rm -rf ${B}/.f2003
	@echo "strict f2003: all clean"

clean :
	rm -rf ${B}

.PHONY : default clean lib shared mpi f2003check serial.out embed_example.out \
	mlff_example.out train_example.out libverify_example.out uq_example.out \
	customloss_example.out hod_ff_example.out roofline.out gen_c_ref.out fdcheck.out \
	gen_hod.out gen_pinn.out c_example.out c_train_example.out
