"""

; ModuleID = 'func4'
source_filename = "<string>"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin24.3.0"

@.const.func4 = internal constant [6 x i8] c"func4\00"
@_ZN08NumbaEnv8__main__5func4B2v9B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx = common local_unnamed_addr global i8* null
@".const.missing Environment: _ZN08NumbaEnv8__main__5func4B2v9B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx" = internal constant [97 x i8] c"missing Environment: _ZN08NumbaEnv8__main__5func4B2v9B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx\00"
@".const.Error creating Python tuple from runtime exception arguments" = internal constant [61 x i8] c"Error creating Python tuple from runtime exception arguments\00"
@".const.unknown error when calling native function" = internal constant [43 x i8] c"unknown error when calling native function\00"
@".const.Error creating Python tuple from runtime exception arguments.1" = internal constant [61 x i8] c"Error creating Python tuple from runtime exception arguments\00"
@".const.unknown error when calling native function.2" = internal constant [43 x i8] c"unknown error when calling native function\00"
@".const.<numba.core.cpu.CPUContext object at 0x110f389b0>" = internal constant [50 x i8] c"<numba.core.cpu.CPUContext object at 0x110f389b0>\00"
@_ZN08NumbaEnv8__main__5func3B3v10B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx = common local_unnamed_addr global i8* null
@.const.pickledata.4575674496 = internal constant [77 x i8] c"\80\04\95B\00\00\00\00\00\00\00\8C\08builtins\94\8C\0AValueError\94\93\94\8C\1Fnegative dimensions not allowed\94\85\94N\87\94."
@.const.pickledata.4575674496.sha1 = internal constant [20 x i8] c"3\1B\85c\BD\B9\DA\C8\1B8B\22s\05,Ho\C1pk"
@.const.picklebuf.4575674496 = internal constant { i8*, i32, i8*, i8*, i32 } { i8* getelementptr inbounds ([77 x i8], [77 x i8]* @.const.pickledata.4575674496, i32 0, i32 0), i32 77, i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.const.pickledata.4575674496.sha1, i32 0, i32 0), i8* null, i32 0 }
@.const.pickledata.4575674432 = internal constant [137 x i8] c"\80\04\95~\00\00\00\00\00\00\00\8C\08builtins\94\8C\0AValueError\94\93\94\8C[array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.\94\85\94N\87\94."
@.const.pickledata.4575674432.sha1 = internal constant [20 x i8] c"X\E1N\CC\B5\07\B1\E0 i\81t\02#\E6\85\CB\8C<W"
@.const.picklebuf.4575674432 = internal constant { i8*, i32, i8*, i8*, i32 } { i8* getelementptr inbounds ([137 x i8], [137 x i8]* @.const.pickledata.4575674432, i32 0, i32 0), i32 137, i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.const.pickledata.4575674432.sha1, i32 0, i32 0), i8* null, i32 0 }
@.const.pickledata.4575216064 = internal constant [86 x i8] c"\80\04\95K\00\00\00\00\00\00\00\8C\08builtins\94\8C\0BMemoryError\94\93\94\8C'Allocation failed (probably too large).\94\85\94N\87\94."
@.const.pickledata.4575216064.sha1 = internal constant [20 x i8] c"\BA(\9D\81\F0\\p \F3G|\15sH\04\DFe\AB\E2\09"
@.const.picklebuf.4575216064 = internal constant { i8*, i32, i8*, i8*, i32 } { i8* getelementptr inbounds ([86 x i8], [86 x i8]* @.const.pickledata.4575216064, i32 0, i32 0), i32 86, i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.const.pickledata.4575216064.sha1, i32 0, i32 0), i8* null, i32 0 }
@.const.pickledata.4575674496.14 = internal constant [77 x i8] c"\80\04\95B\00\00\00\00\00\00\00\8C\08builtins\94\8C\0AValueError\94\93\94\8C\1Fnegative dimensions not allowed\94\85\94N\87\94."
@.const.pickledata.4575674496.sha1.15 = internal constant [20 x i8] c"3\1B\85c\BD\B9\DA\C8\1B8B\22s\05,Ho\C1pk"
@.const.picklebuf.4575674496.13 = internal constant { i8*, i32, i8*, i8*, i32 } { i8* getelementptr inbounds ([77 x i8], [77 x i8]* @.const.pickledata.4575674496.14, i32 0, i32 0), i32 77, i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.const.pickledata.4575674496.sha1.15, i32 0, i32 0), i8* null, i32 0 }
@.const.pickledata.4575674432.16 = internal constant [137 x i8] c"\80\04\95~\00\00\00\00\00\00\00\8C\08builtins\94\8C\0AValueError\94\93\94\8C[array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.\94\85\94N\87\94."
@.const.pickledata.4575674432.sha1.17 = internal constant [20 x i8] c"X\E1N\CC\B5\07\B1\E0 i\81t\02#\E6\85\CB\8C<W"
@.const.picklebuf.4575674432.12 = internal constant { i8*, i32, i8*, i8*, i32 } { i8* getelementptr inbounds ([137 x i8], [137 x i8]* @.const.pickledata.4575674432.16, i32 0, i32 0), i32 137, i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.const.pickledata.4575674432.sha1.17, i32 0, i32 0), i8* null, i32 0 }
@.const.pickledata.4575216064.18 = internal constant [86 x i8] c"\80\04\95K\00\00\00\00\00\00\00\8C\08builtins\94\8C\0BMemoryError\94\93\94\8C'Allocation failed (probably too large).\94\85\94N\87\94."
@.const.pickledata.4575216064.sha1.19 = internal constant [20 x i8] c"\BA(\9D\81\F0\\p \F3G|\15sH\04\DFe\AB\E2\09"
@.const.picklebuf.4575216064.11 = internal constant { i8*, i32, i8*, i8*, i32 } { i8* getelementptr inbounds ([86 x i8], [86 x i8]* @.const.pickledata.4575216064.18, i32 0, i32 0), i32 86, i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.const.pickledata.4575216064.sha1.19, i32 0, i32 0), i8* null, i32 0 }
@.const.pickledata.4575674496.16 = internal constant [77 x i8] c"\80\04\95B\00\00\00\00\00\00\00\8C\08builtins\94\8C\0AValueError\94\93\94\8C\1Fnegative dimensions not allowed\94\85\94N\87\94."
@.const.pickledata.4575674496.sha1.17 = internal constant [20 x i8] c"3\1B\85c\BD\B9\DA\C8\1B8B\22s\05,Ho\C1pk"
@.const.picklebuf.4575674496.15 = internal constant { i8*, i32, i8*, i8*, i32 } { i8* getelementptr inbounds ([77 x i8], [77 x i8]* @.const.pickledata.4575674496.16, i32 0, i32 0), i32 77, i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.const.pickledata.4575674496.sha1.17, i32 0, i32 0), i8* null, i32 0 }
@.const.pickledata.4575674432.18 = internal constant [137 x i8] c"\80\04\95~\00\00\00\00\00\00\00\8C\08builtins\94\8C\0AValueError\94\93\94\8C[array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.\94\85\94N\87\94."
@.const.pickledata.4575674432.sha1.19 = internal constant [20 x i8] c"X\E1N\CC\B5\07\B1\E0 i\81t\02#\E6\85\CB\8C<W"
@.const.picklebuf.4575674432.14 = internal constant { i8*, i32, i8*, i8*, i32 } { i8* getelementptr inbounds ([137 x i8], [137 x i8]* @.const.pickledata.4575674432.18, i32 0, i32 0), i32 137, i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.const.pickledata.4575674432.sha1.19, i32 0, i32 0), i8* null, i32 0 }
@.const.pickledata.4575216064.20 = internal constant [86 x i8] c"\80\04\95K\00\00\00\00\00\00\00\8C\08builtins\94\8C\0BMemoryError\94\93\94\8C'Allocation failed (probably too large).\94\85\94N\87\94."
@.const.pickledata.4575216064.sha1.21 = internal constant [20 x i8] c"\BA(\9D\81\F0\\p \F3G|\15sH\04\DFe\AB\E2\09"
@.const.picklebuf.4575216064.13 = internal constant { i8*, i32, i8*, i8*, i32 } { i8* getelementptr inbounds ([86 x i8], [86 x i8]* @.const.pickledata.4575216064.20, i32 0, i32 0), i32 86, i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.const.pickledata.4575216064.sha1.21, i32 0, i32 0), i8* null, i32 0 }
@_ZN08NumbaEnv8__main__5func2B3v11B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx = common local_unnamed_addr global i8* null
@_Py_NoneStruct = external global i8
@PyExc_StopIteration = external global i8
@_ZN08NumbaEnv5numba7cpython11old_numbers14int_power_impl12_3clocals_3e9int_powerB3v12B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAExx = common local_unnamed_addr global i8* null
@.const.pickledata.4575674496.18 = internal constant [77 x i8] c"\80\04\95B\00\00\00\00\00\00\00\8C\08builtins\94\8C\0AValueError\94\93\94\8C\1Fnegative dimensions not allowed\94\85\94N\87\94."
@.const.pickledata.4575674496.sha1.19 = internal constant [20 x i8] c"3\1B\85c\BD\B9\DA\C8\1B8B\22s\05,Ho\C1pk"
@.const.picklebuf.4575674496.17 = internal constant { i8*, i32, i8*, i8*, i32 } { i8* getelementptr inbounds ([77 x i8], [77 x i8]* @.const.pickledata.4575674496.18, i32 0, i32 0), i32 77, i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.const.pickledata.4575674496.sha1.19, i32 0, i32 0), i8* null, i32 0 }
@.const.pickledata.4575674432.20 = internal constant [137 x i8] c"\80\04\95~\00\00\00\00\00\00\00\8C\08builtins\94\8C\0AValueError\94\93\94\8C[array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.\94\85\94N\87\94."
@.const.pickledata.4575674432.sha1.21 = internal constant [20 x i8] c"X\E1N\CC\B5\07\B1\E0 i\81t\02#\E6\85\CB\8C<W"
@.const.picklebuf.4575674432.16 = internal constant { i8*, i32, i8*, i8*, i32 } { i8* getelementptr inbounds ([137 x i8], [137 x i8]* @.const.pickledata.4575674432.20, i32 0, i32 0), i32 137, i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.const.pickledata.4575674432.sha1.21, i32 0, i32 0), i8* null, i32 0 }
@.const.pickledata.4575216064.22 = internal constant [86 x i8] c"\80\04\95K\00\00\00\00\00\00\00\8C\08builtins\94\8C\0BMemoryError\94\93\94\8C'Allocation failed (probably too large).\94\85\94N\87\94."
@.const.pickledata.4575216064.sha1.23 = internal constant [20 x i8] c"\BA(\9D\81\F0\\p \F3G|\15sH\04\DFe\AB\E2\09"
@.const.picklebuf.4575216064.15 = internal constant { i8*, i32, i8*, i8*, i32 } { i8* getelementptr inbounds ([86 x i8], [86 x i8]* @.const.pickledata.4575216064.22, i32 0, i32 0), i32 86, i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.const.pickledata.4575216064.sha1.23, i32 0, i32 0), i8* null, i32 0 }
@_ZN08NumbaEnv8__main__5func1B2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx = common local_unnamed_addr global i8* null
@PyExc_RuntimeError = external global i8
@PyExc_SystemError = external global i8
@_ZN08NumbaEnv5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29 = common local_unnamed_addr global i8* null
@_ZN08NumbaEnv5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29 = common local_unnamed_addr global i8* null
@_ZN08NumbaEnv5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj = common local_unnamed_addr global i8* null
@_ZN08NumbaEnv5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj = common local_unnamed_addr global i8* null
@_ZN08NumbaEnv5numba2np8arrayobj18ol_array_zero_fill12_3clocals_3e4implB2v8B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dE5ArrayIdLi1E1C7mutable7alignedE = common local_unnamed_addr global i8* null
@_ZN08NumbaEnv5numba2np13old_arraymath9array_sum12_3clocals_3e14array_sum_implB3v13B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAE5ArrayIdLi1E1C7mutable7alignedE = common local_unnamed_addr global i8* null
@_ZN08NumbaEnv5numba2np8arrayobj15make_nditer_cls12_3clocals_3e6NdIter13init_specific12_3clocals_3e11check_shapeB3v14B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAE8UniTupleIxLi1EE8UniTupleIxLi1EE = common local_unnamed_addr global i8* null

define i32 @_ZN8__main__5func4B2v9B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx([5 x double]* noalias nocapture writeonly %retptr, { i8*, i32, i8*, i8*, i32 }** noalias nocapture writeonly %excinfo, i64 %arg.m) local_unnamed_addr {
B0.endif:
  %.152 = alloca { double, double, i64, double, double }, align 8
  %excinfo.1 = alloca { i8*, i32, i8*, i8*, i32 }*, align 8
  store { i8*, i32, i8*, i8*, i32 }* null, { i8*, i32, i8*, i8*, i32 }** %excinfo.1, align 8
  %.636 = alloca { double, double, i64, i64, double }, align 8
  %excinfo.3 = alloca { i8*, i32, i8*, i8*, i32 }*, align 8
  store { i8*, i32, i8*, i8*, i32 }* null, { i8*, i32, i8*, i8*, i32 }** %excinfo.3, align 8
  %.848 = alloca { double, double, double, i64, i64 }, align 8
  %excinfo.5 = alloca { i8*, i32, i8*, i8*, i32 }*, align 8
  store { i8*, i32, i8*, i8*, i32 }* null, { i8*, i32, i8*, i8*, i32 }** %excinfo.5, align 8
  %0 = tail call i64 @llvm.smax.i64(i64 %arg.m, i64 0)
  %.10484.not = icmp slt i64 %arg.m, 1
  br i1 %.10484.not, label %B228.else.if, label %B54.preheader

B54.preheader:                                    ; preds = %B0.endif
  br label %B54

B54:                                              ; preds = %B54.preheader, %B192.else.if
  %lsr.iv125 = phi i64 [ 10, %B54.preheader ], [ %lsr.iv.next126, %B192.else.if ]
  %lsr.iv123 = phi i64 [ 0, %B54.preheader ], [ %lsr.iv.next124, %B192.else.if ]
  %total_sum.4.090 = phi double [ %total_sum.5.1, %B192.else.if ], [ 0.000000e+00, %B54.preheader ]
  %max_value.3.089 = phi double [ %max_value.4.1, %B192.else.if ], [ -1.000000e+07, %B54.preheader ]
  %factorial_sum.3.088 = phi double [ %factorial_sum.4.1, %B192.else.if ], [ 0.000000e+00, %B54.preheader ]
  %count_total.3.087 = phi double [ %count_total.4.1, %B192.else.if ], [ 0.000000e+00, %B54.preheader ]
  %.46.085 = phi i64 [ %.117, %B192.else.if ], [ 1, %B54.preheader ]
  %1 = bitcast { double, double, i64, double, double }* %.152 to i8*
  %2 = udiv i64 %.46.085, 11
  %3 = mul nuw nsw i64 %2, 11
  %4 = add nsw i64 %3, -1
  %5 = udiv i64 %.46.085, 7
  %6 = mul nuw nsw i64 %5, 7
  %7 = add nsw i64 %6, -1
  %8 = udiv i64 %.46.085, 5
  %9 = mul nuw nsw i64 %8, 5
  %10 = add nsw i64 %9, -1
  %11 = add i64 %lsr.iv123, 1
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(40) %1, i8 0, i64 40, i1 false)
  %.156 = call i32 @_ZN8__main__5func3B3v10B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx({ double, double, i64, double, double }* nonnull %.152, { i8*, i32, i8*, i8*, i32 }** nonnull %excinfo.1, i64 %lsr.iv125)
  switch i32 %.156, label %B54.if [
    i32 -2, label %B54.endif
    i32 0, label %B54.endif
  ]

B270:                                             ; preds = %B270.preheader, %B270.endif
  %lsr.iv121 = phi i64 [ 5, %B270.preheader ], [ %lsr.iv.next122, %B270.endif ]
  %lsr.iv119 = phi i64 [ %spec.select66, %B270.preheader ], [ %lsr.iv.next120, %B270.endif ]
  %max_value.5.082 = phi double [ %.676, %B270.endif ], [ %max_value.3.0.lcssa, %B270.preheader ]
  %total_sum.6.081 = phi double [ %.671, %B270.endif ], [ %total_sum.4.0.lcssa, %B270.preheader ]
  %12 = bitcast { double, double, i64, i64, double }* %.636 to i8*
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(40) %12, i8 0, i64 40, i1 false)
  %.640 = call i32 @_ZN8__main__5func2B3v11B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx({ double, double, i64, i64, double }* nonnull %.636, { i8*, i32, i8*, i8*, i32 }** nonnull %excinfo.3, i64 %lsr.iv121)
  switch i32 %.640, label %B270.if [
    i32 -2, label %B270.endif
    i32 0, label %B270.endif
  ]

B348:                                             ; preds = %B270.endif, %B228.else.if
  %total_sum.6.0.lcssa = phi double [ %total_sum.4.0.lcssa, %B228.else.if ], [ %.671, %B270.endif ]
  %max_value.5.0.lcssa = phi double [ %max_value.3.0.lcssa, %B228.else.if ], [ %.676, %B270.endif ]
  %.15.i.i = icmp slt i64 %arg.m, 0
  br i1 %.15.i.i, label %B348.if, label %B0.endif.endif.i.i, !prof !0

B0.endif.endif.i.i:                               ; preds = %B348
  %.29.i.i = tail call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %arg.m, i64 8)
  %.31.i.i = extractvalue { i64, i1 } %.29.i.i, 1
  br i1 %.31.i.i, label %B348.if, label %B0.endif.endif.endif.i.i, !prof !0

B0.endif.endif.endif.i.i:                         ; preds = %B0.endif.endif.i.i
  %.30.i.i = extractvalue { i64, i1 } %.29.i.i, 0
  %.7.i.i.i.i = tail call i8* @NRT_MemInfo_alloc_aligned(i64 %.30.i.i, i32 32), !noalias !1
  %.8.i.i.i.i = icmp eq i8* %.7.i.i.i.i, null
  br i1 %.8.i.i.i.i, label %B348.if, label %B348.endif.endif, !prof !0

B418:                                             ; preds = %B418.preheader, %B418.endif
  %.752.076 = phi i64 [ %.847, %B418.endif ], [ 0, %B418.preheader ]
  %13 = bitcast { double, double, double, i64, i64 }* %.848 to i8*
  %.847 = add nuw nsw i64 %.752.076, 1
  %14 = add i64 %.752.076, 1
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(40) %13, i8 0, i64 40, i1 false)
  %.852 = call i32 @_ZN8__main__5func1B2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx({ double, double, double, i64, i64 }* nonnull %.848, { i8*, i32, i8*, i8*, i32 }** nonnull %excinfo.5, i64 %14)
  switch i32 %.852, label %B418.if [
    i32 -2, label %B418.endif
    i32 0, label %B418.endif
  ]

B52.endif.i.preheader:                            ; preds = %B418.endif
  %smax = call i64 @llvm.smax.i64(i64 %arg.m, i64 1)
  br label %B52.endif.i

B52.endif.i:                                      ; preds = %B52.endif.i.preheader, %B52.endif.i
  %lsr.iv117 = phi double* [ %.6.i1.i.i, %B52.endif.i.preheader ], [ %scevgep, %B52.endif.i ]
  %lsr.iv = phi i64 [ %smax, %B52.endif.i.preheader ], [ %lsr.iv.next, %B52.endif.i ]
  %c.2.06.i = phi double [ %.239.i, %B52.endif.i ], [ 0.000000e+00, %B52.endif.i.preheader ]
  %.236.i = load double, double* %lsr.iv117, align 8, !noalias !14
  %.239.i = fadd double %c.2.06.i, %.236.i
  %lsr.iv.next = add nsw i64 %lsr.iv, -1
  %scevgep = getelementptr double, double* %lsr.iv117, i64 1
  %exitcond.not = icmp eq i64 %lsr.iv.next, 0
  br i1 %exitcond.not, label %B472.endif, label %B52.endif.i, !prof !0

common.ret:                                       ; preds = %B418.if, %B348.if, %B270.if, %B54.if, %B472.endif
  %common.ret.op = phi i32 [ 0, %B472.endif ], [ 1, %B348.if ], [ %.852, %B418.if ], [ %.640, %B270.if ], [ %.156, %B54.if ]
  ret i32 %common.ret.op

B54.if:                                           ; preds = %B54
  %.164 = icmp sgt i32 %.156, 0
  %.157 = load { i8*, i32, i8*, i8*, i32 }*, { i8*, i32, i8*, i8*, i32 }** %excinfo.1, align 8
  %.165 = select i1 %.164, { i8*, i32, i8*, i8*, i32 }* %.157, { i8*, i32, i8*, i8*, i32 }* undef
  store { i8*, i32, i8*, i8*, i32 }* %.165, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  br label %common.ret

B54.endif:                                        ; preds = %B54, %B54
  %15 = bitcast { double, double, i64, double, double }* %.152 to double*
  %16 = bitcast { double, double, i64, double, double }* %.152 to i8*
  %sunkaddr = getelementptr inbounds i8, i8* %16, i64 32
  %17 = bitcast i8* %sunkaddr to double*
  %.166.fca.4.load = load double, double* %17, align 8
  %18 = bitcast { double, double, i64, double, double }* %.152 to i8*
  %sunkaddr130 = getelementptr inbounds i8, i8* %18, i64 24
  %19 = bitcast i8* %sunkaddr130 to double*
  %.166.fca.3.load = load double, double* %19, align 8
  %20 = bitcast { double, double, i64, double, double }* %.152 to i8*
  %sunkaddr131 = getelementptr inbounds i8, i8* %20, i64 8
  %21 = bitcast i8* %sunkaddr131 to double*
  %.166.fca.1.load = load double, double* %21, align 8
  %.166.fca.0.load = load double, double* %15, align 8
  %.196 = fadd double %total_sum.4.090, %.166.fca.0.load
  %.204 = fcmp ogt double %.166.fca.1.load, %max_value.3.089
  %max_value.4.1 = select i1 %.204, double %.166.fca.1.load, double %max_value.3.089
  %.222 = fadd double %factorial_sum.3.088, %.166.fca.3.load
  %.232 = fadd double %count_total.3.087, %.166.fca.4.load
  %.283 = icmp eq i64 %10, %lsr.iv123
  %.286 = fmul double %.196, 0x3FEE666666666666
  %total_sum.5.1 = select i1 %.283, double %.286, double %.196
  %.341 = icmp eq i64 %7, %lsr.iv123
  br i1 %.341, label %B182.else, label %B192.else.if

B182.else:                                        ; preds = %B54.endif
  %.11.i = frem double %.222, 2.000000e+00
  %.12.i = fsub double %.222, %.11.i
  %.13.i = fmul double %.12.i, 5.000000e-01
  %22 = fcmp olt double %.11.i, 0.000000e+00
  %.22.i = fadd double %.13.i, -1.000000e+00
  %.7.0.i = select i1 %22, double %.22.i, double %.13.i, !prof !17
  %.32.i = fcmp ueq double %.7.0.i, 0.000000e+00
  br i1 %.32.i, label %entry.endif.endif.if.i, label %entry.endif.endif.thread.i

entry.endif.endif.thread.i:                       ; preds = %B182.else
  %.34.i = tail call double @llvm.floor.f64(double %.7.0.i)
  %.35.i = fsub double %.7.0.i, %.34.i
  %.36.i = fadd double %.34.i, 1.000000e+00
  %.37.i = fcmp ogt double %.35.i, 5.000000e-01
  %.38.i = select i1 %.37.i, double %.36.i, double %.34.i
  br label %B192.else.if

entry.endif.endif.if.i:                           ; preds = %B182.else
  %.43.i = fmul double %.7.0.i, %.7.0.i
  %.45.i = fmul double %.222, %.43.i
  %.46.i = fmul double %.45.i, 5.000000e-01
  br label %B192.else.if

B192.else.if:                                     ; preds = %B54.endif, %entry.endif.endif.thread.i, %entry.endif.endif.if.i
  %factorial_sum.4.1 = phi double [ %.222, %B54.endif ], [ %.46.i, %entry.endif.endif.if.i ], [ %.38.i, %entry.endif.endif.thread.i ]
  %.411 = icmp eq i64 %4, %lsr.iv123
  %.191.i = mul nsw i64 %11, %11
  %.450 = sitofp i64 %.191.i to double
  %.451 = select i1 %.411, double %.450, double -0.000000e+00
  %count_total.4.1 = fadd double %.232, %.451
  %lsr.iv.next124 = add nuw nsw i64 %lsr.iv123, 1
  %lsr.iv.next126 = add i64 %lsr.iv125, 10
  %.117 = add nuw i64 %.46.085, 1
  %exitcond100.not = icmp eq i64 %0, %lsr.iv.next124
  br i1 %exitcond100.not, label %B228.else.if, label %B54

B228.else.if:                                     ; preds = %B192.else.if, %B0.endif
  %count_total.3.0.lcssa = phi double [ 0.000000e+00, %B0.endif ], [ %count_total.4.1, %B192.else.if ]
  %factorial_sum.3.0.lcssa = phi double [ 0.000000e+00, %B0.endif ], [ %factorial_sum.4.1, %B192.else.if ]
  %max_value.3.0.lcssa = phi double [ -1.000000e+07, %B0.endif ], [ %max_value.4.1, %B192.else.if ]
  %total_sum.4.0.lcssa = phi double [ 0.000000e+00, %B0.endif ], [ %total_sum.5.1, %B192.else.if ]
  %.494 = sdiv i64 %arg.m, 2
  %23 = and i64 %arg.m, -9223372036854775807
  %.497 = icmp eq i64 %23, -9223372036854775807
  %.504 = sext i1 %.497 to i64
  %spec.select66 = add nsw i64 %.494, %.504
  %.59678.not = icmp slt i64 %spec.select66, 1
  br i1 %.59678.not, label %B348, label %B270.preheader

B270.preheader:                                   ; preds = %B228.else.if
  br label %B270

B270.if:                                          ; preds = %B270
  %.648 = icmp sgt i32 %.640, 0
  %.641 = load { i8*, i32, i8*, i8*, i32 }*, { i8*, i32, i8*, i8*, i32 }** %excinfo.3, align 8
  %.649 = select i1 %.648, { i8*, i32, i8*, i8*, i32 }* %.641, { i8*, i32, i8*, i8*, i32 }* undef
  store { i8*, i32, i8*, i8*, i32 }* %.649, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  br label %common.ret

B270.endif:                                       ; preds = %B270, %B270
  %24 = bitcast { double, double, i64, i64, double }* %.636 to double*
  %25 = bitcast { double, double, i64, i64, double }* %.636 to i8*
  %sunkaddr132 = getelementptr inbounds i8, i8* %25, i64 8
  %26 = bitcast i8* %sunkaddr132 to double*
  %.650.fca.1.load = load double, double* %26, align 8
  %.650.fca.0.load = load double, double* %24, align 8
  %.671 = fadd double %total_sum.6.081, %.650.fca.0.load
  %.675 = fcmp ogt double %.650.fca.1.load, %max_value.5.082
  %.676 = select i1 %.675, double %.650.fca.1.load, double %max_value.5.082
  %lsr.iv.next120 = add i64 %lsr.iv119, -1
  %lsr.iv.next122 = add i64 %lsr.iv121, 5
  %exitcond99.not = icmp eq i64 %lsr.iv.next120, 0
  br i1 %exitcond99.not, label %B348, label %B270

B348.if:                                          ; preds = %B0.endif.endif.endif.i.i, %B0.endif.endif.i.i, %B348
  %excinfo.1.0.ph.i = phi { i8*, i32, i8*, i8*, i32 }* [ @.const.picklebuf.4575216064.15, %B0.endif.endif.endif.i.i ], [ @.const.picklebuf.4575674432.16, %B0.endif.endif.i.i ], [ @.const.picklebuf.4575674496.17, %B348 ]
  store { i8*, i32, i8*, i8*, i32 }* %excinfo.1.0.ph.i, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  br label %common.ret

B348.endif.endif:                                 ; preds = %B0.endif.endif.endif.i.i
  %27 = icmp slt i64 %arg.m, 1
  %.5.i.i.i = getelementptr i8, i8* %.7.i.i.i.i, i64 24
  %28 = bitcast i8* %.5.i.i.i to double**
  %.6.i1.i.i = load double*, double** %28, align 8, !noalias !18
  %.26.i.i = shl nuw nsw i64 %arg.m, 3
  %.27.i.i = bitcast double* %.6.i1.i.i to i8*
  tail call void @llvm.memset.p0i8.i64(i8* align 1 %.27.i.i, i8 0, i64 %.26.i.i, i1 false), !noalias !19
  br i1 %27, label %B472.endif, label %B418.preheader

B418.preheader:                                   ; preds = %B348.endif.endif
  br label %B418

B418.if:                                          ; preds = %B418
  %.860 = icmp sgt i32 %.852, 0
  %.853 = load { i8*, i32, i8*, i8*, i32 }*, { i8*, i32, i8*, i8*, i32 }** %excinfo.5, align 8
  %.861 = select i1 %.860, { i8*, i32, i8*, i8*, i32 }* %.853, { i8*, i32, i8*, i8*, i32 }* undef
  store { i8*, i32, i8*, i8*, i32 }* %.861, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  br label %common.ret

B418.endif:                                       ; preds = %B418, %B418
  %29 = bitcast { double, double, double, i64, i64 }* %.848 to double*
  %.862.fca.0.load = load double, double* %29, align 8
  %scevgep118 = getelementptr double, double* %.6.i1.i.i, i64 %.752.076
  store double %.862.fca.0.load, double* %scevgep118, align 8
  %exitcond98.not = icmp eq i64 %0, %.847
  br i1 %exitcond98.not, label %B52.endif.i.preheader, label %B418

B472.endif:                                       ; preds = %B52.endif.i, %B348.endif.endif
  %c.2.0.lcssa.i = phi double [ 0.000000e+00, %B348.endif.endif ], [ %.239.i, %B52.endif.i ]
  tail call void @NRT_decref(i8* nonnull %.7.i.i.i.i)
  %retptr.repack133 = bitcast [5 x double]* %retptr to double*
  store double %total_sum.6.0.lcssa, double* %retptr.repack133, align 8
  %retptr.repack28 = getelementptr inbounds [5 x double], [5 x double]* %retptr, i64 0, i64 1
  store double %max_value.5.0.lcssa, double* %retptr.repack28, align 8
  %retptr.repack30 = getelementptr inbounds [5 x double], [5 x double]* %retptr, i64 0, i64 2
  store double %factorial_sum.3.0.lcssa, double* %retptr.repack30, align 8
  %retptr.repack32 = getelementptr inbounds [5 x double], [5 x double]* %retptr, i64 0, i64 3
  store double %count_total.3.0.lcssa, double* %retptr.repack32, align 8
  %retptr.repack34 = getelementptr inbounds [5 x double], [5 x double]* %retptr, i64 0, i64 4
  store double %c.2.0.lcssa.i, double* %retptr.repack34, align 8
  br label %common.ret
}

define i8* @_ZN7cpython8__main__5func4B2v9B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx(i8* nocapture readnone %py_closure, i8* %py_args, i8* nocapture readnone %py_kws) local_unnamed_addr {
entry:
  %.5 = alloca i8*, align 8
  %.6 = call i32 (i8*, i8*, i64, i64, ...) @PyArg_UnpackTuple(i8* %py_args, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.const.func4, i64 0, i64 0), i64 1, i64 1, i8** nonnull %.5)
  %.7 = icmp eq i32 %.6, 0
  %.36 = alloca [5 x double], align 8
  %excinfo = alloca { i8*, i32, i8*, i8*, i32 }*, align 8
  store { i8*, i32, i8*, i8*, i32 }* null, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  br i1 %.7, label %common.ret, label %entry.endif, !prof !0

common.ret:                                       ; preds = %entry.endif.endif.endif.endif.endif.endif, %entry.endif.endif.endif.endif.endif.if.endif, %entry.endif.endif.endif.endif.endif.if.endif.if, %entry.endif.endif.endif.endif.endif.endif.if, %entry.endif.endif.endif.endif.endif.endif.endif.endif, %entry.endif.endif.endif, %entry, %entry.endif.endif.endif.endif.endif.if.if.if, %entry.endif.endif.endif.endif.if.endif, %entry.endif.endif.endif.endif.if.if, %entry.endif.if
  %common.ret.op = phi i8* [ null, %entry.endif.if ], [ @_Py_NoneStruct, %entry.endif.endif.endif.endif.if.if ], [ %.55, %entry.endif.endif.endif.endif.if.endif ], [ null, %entry.endif.endif.endif.endif.endif.if.if.if ], [ null, %entry ], [ null, %entry.endif.endif.endif ], [ null, %entry.endif.endif.endif.endif.endif.endif.endif.endif ], [ null, %entry.endif.endif.endif.endif.endif.endif.if ], [ null, %entry.endif.endif.endif.endif.endif.if.endif.if ], [ null, %entry.endif.endif.endif.endif.endif.if.endif ], [ null, %entry.endif.endif.endif.endif.endif.endif ]
  ret i8* %common.ret.op

entry.endif:                                      ; preds = %entry
  %.11 = load i8*, i8** @_ZN08NumbaEnv8__main__5func4B2v9B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx, align 8
  %.16 = icmp eq i8* %.11, null
  br i1 %.16, label %entry.endif.if, label %entry.endif.endif, !prof !0

entry.endif.if:                                   ; preds = %entry.endif
  call void @PyErr_SetString(i8* nonnull @PyExc_RuntimeError, i8* getelementptr inbounds ([97 x i8], [97 x i8]* @".const.missing Environment: _ZN08NumbaEnv8__main__5func4B2v9B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx", i64 0, i64 0))
  br label %common.ret

entry.endif.endif:                                ; preds = %entry.endif
  %.20 = load i8*, i8** %.5, align 8
  %.23 = call i8* @PyNumber_Long(i8* %.20)
  %.24.not = icmp eq i8* %.23, null
  br i1 %.24.not, label %entry.endif.endif.endif, label %entry.endif.endif.if, !prof !0

entry.endif.endif.if:                             ; preds = %entry.endif.endif
  %.26 = call i64 @PyLong_AsLongLong(i8* nonnull %.23)
  call void @Py_DecRef(i8* nonnull %.23)
  br label %entry.endif.endif.endif

entry.endif.endif.endif:                          ; preds = %entry.endif.endif.if, %entry.endif.endif
  %.21.0 = phi i64 [ %.26, %entry.endif.endif.if ], [ 0, %entry.endif.endif ]
  %.31 = call i8* @PyErr_Occurred()
  %.32.not = icmp eq i8* %.31, null
  br i1 %.32.not, label %entry.endif.endif.endif.endif, label %common.ret, !prof !22

entry.endif.endif.endif.endif:                    ; preds = %entry.endif.endif.endif
  %0 = bitcast [5 x double]* %.36 to i8*
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(40) %0, i8 0, i64 40, i1 false)
  %.40 = call i32 @_ZN8__main__5func4B2v9B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx([5 x double]* nonnull %.36, { i8*, i32, i8*, i8*, i32 }** nonnull %excinfo, i64 %.21.0) #2
  %.41 = load { i8*, i32, i8*, i8*, i32 }*, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  %.48 = icmp sgt i32 %.40, 0
  %.49 = select i1 %.48, { i8*, i32, i8*, i8*, i32 }* %.41, { i8*, i32, i8*, i8*, i32 }* undef
  switch i32 %.40, label %entry.endif.endif.endif.endif.endif [
    i32 -2, label %entry.endif.endif.endif.endif.if.if
    i32 0, label %entry.endif.endif.endif.endif.if.endif
  ]

entry.endif.endif.endif.endif.endif:              ; preds = %entry.endif.endif.endif.endif
  %1 = icmp sgt i32 %.40, 0
  br i1 %1, label %entry.endif.endif.endif.endif.endif.if, label %entry.endif.endif.endif.endif.endif.endif

entry.endif.endif.endif.endif.if.if:              ; preds = %entry.endif.endif.endif.endif
  call void @Py_IncRef(i8* nonnull @_Py_NoneStruct)
  br label %common.ret

entry.endif.endif.endif.endif.if.endif:           ; preds = %entry.endif.endif.endif.endif
  %2 = bitcast [5 x double]* %.36 to double*
  %3 = bitcast [5 x double]* %.36 to i8*
  %sunkaddr = getelementptr inbounds i8, i8* %3, i64 32
  %4 = bitcast i8* %sunkaddr to double*
  %.50.fca.4.load = load double, double* %4, align 8
  %5 = bitcast [5 x double]* %.36 to i8*
  %sunkaddr2 = getelementptr inbounds i8, i8* %5, i64 24
  %6 = bitcast i8* %sunkaddr2 to double*
  %.50.fca.3.load = load double, double* %6, align 8
  %7 = bitcast [5 x double]* %.36 to i8*
  %sunkaddr3 = getelementptr inbounds i8, i8* %7, i64 16
  %8 = bitcast i8* %sunkaddr3 to double*
  %.50.fca.2.load = load double, double* %8, align 8
  %9 = bitcast [5 x double]* %.36 to i8*
  %sunkaddr4 = getelementptr inbounds i8, i8* %9, i64 8
  %10 = bitcast i8* %sunkaddr4 to double*
  %.50.fca.1.load = load double, double* %10, align 8
  %.50.fca.0.load = load double, double* %2, align 8
  %.55 = call i8* @PyTuple_New(i64 5)
  %.57 = call i8* @PyFloat_FromDouble(double %.50.fca.0.load)
  %.58 = call i32 @PyTuple_SetItem(i8* %.55, i64 0, i8* %.57)
  %.60 = call i8* @PyFloat_FromDouble(double %.50.fca.1.load)
  %.61 = call i32 @PyTuple_SetItem(i8* %.55, i64 1, i8* %.60)
  %.63 = call i8* @PyFloat_FromDouble(double %.50.fca.2.load)
  %.64 = call i32 @PyTuple_SetItem(i8* %.55, i64 2, i8* %.63)
  %.66 = call i8* @PyFloat_FromDouble(double %.50.fca.3.load)
  %.67 = call i32 @PyTuple_SetItem(i8* %.55, i64 3, i8* %.66)
  %.69 = call i8* @PyFloat_FromDouble(double %.50.fca.4.load)
  %.70 = call i32 @PyTuple_SetItem(i8* %.55, i64 4, i8* %.69)
  br label %common.ret

entry.endif.endif.endif.endif.endif.if:           ; preds = %entry.endif.endif.endif.endif.endif
  call void @PyErr_Clear()
  %.75 = load { i8*, i32, i8*, i8*, i32 }, { i8*, i32, i8*, i8*, i32 }* %.49, align 8
  %.76 = extractvalue { i8*, i32, i8*, i8*, i32 } %.75, 4
  %.77 = icmp sgt i32 %.76, 0
  %.80 = extractvalue { i8*, i32, i8*, i8*, i32 } %.75, 0
  %.82 = extractvalue { i8*, i32, i8*, i8*, i32 } %.75, 1
  br i1 %.77, label %entry.endif.endif.endif.endif.endif.if.if, label %entry.endif.endif.endif.endif.endif.if.else

entry.endif.endif.endif.endif.endif.endif:        ; preds = %entry.endif.endif.endif.endif.endif
  switch i32 %.40, label %entry.endif.endif.endif.endif.endif.endif.endif.endif [
    i32 -3, label %entry.endif.endif.endif.endif.endif.endif.if
    i32 -1, label %common.ret
  ]

entry.endif.endif.endif.endif.endif.if.if:        ; preds = %entry.endif.endif.endif.endif.endif.if
  %.83 = sext i32 %.82 to i64
  %.84 = call i8* @PyBytes_FromStringAndSize(i8* %.80, i64 %.83)
  %.85 = load { i8*, i32, i8*, i8*, i32 }, { i8*, i32, i8*, i8*, i32 }* %.49, align 8
  %.86 = extractvalue { i8*, i32, i8*, i8*, i32 } %.85, 2
  %.88 = extractvalue { i8*, i32, i8*, i8*, i32 } %.85, 3
  %.89 = bitcast i8* %.88 to i8* (i8*)*
  %.90 = call i8* %.89(i8* %.86)
  %.91 = icmp eq i8* %.90, null
  br i1 %.91, label %entry.endif.endif.endif.endif.endif.if.if.if, label %entry.endif.endif.endif.endif.endif.if.if.endif, !prof !0

entry.endif.endif.endif.endif.endif.if.else:      ; preds = %entry.endif.endif.endif.endif.endif.if
  %.104 = extractvalue { i8*, i32, i8*, i8*, i32 } %.75, 2
  %.105 = call i8* @numba_unpickle(i8* %.80, i32 %.82, i8* %.104)
  br label %entry.endif.endif.endif.endif.endif.if.endif

entry.endif.endif.endif.endif.endif.if.endif:     ; preds = %entry.endif.endif.endif.endif.endif.if.if.endif, %entry.endif.endif.endif.endif.endif.if.else
  %.107 = phi i8* [ %.95, %entry.endif.endif.endif.endif.endif.if.if.endif ], [ %.105, %entry.endif.endif.endif.endif.endif.if.else ]
  %.108.not = icmp eq i8* %.107, null
  br i1 %.108.not, label %common.ret, label %entry.endif.endif.endif.endif.endif.if.endif.if, !prof !0

entry.endif.endif.endif.endif.endif.if.if.if:     ; preds = %entry.endif.endif.endif.endif.endif.if.if
  call void @PyErr_SetString(i8* nonnull @PyExc_RuntimeError, i8* getelementptr inbounds ([61 x i8], [61 x i8]* @".const.Error creating Python tuple from runtime exception arguments", i64 0, i64 0))
  br label %common.ret

entry.endif.endif.endif.endif.endif.if.if.endif:  ; preds = %entry.endif.endif.endif.endif.endif.if.if
  %.95 = call i8* @numba_runtime_build_excinfo_struct(i8* %.84, i8* nonnull %.90)
  %.96 = bitcast { i8*, i32, i8*, i8*, i32 }* %.49 to i8*
  call void @NRT_Free(i8* nonnull %.96)
  br label %entry.endif.endif.endif.endif.endif.if.endif

entry.endif.endif.endif.endif.endif.if.endif.if:  ; preds = %entry.endif.endif.endif.endif.endif.if.endif
  call void @numba_do_raise(i8* nonnull %.107)
  br label %common.ret

entry.endif.endif.endif.endif.endif.endif.if:     ; preds = %entry.endif.endif.endif.endif.endif.endif
  call void @PyErr_SetNone(i8* nonnull @PyExc_StopIteration)
  br label %common.ret

entry.endif.endif.endif.endif.endif.endif.endif.endif: ; preds = %entry.endif.endif.endif.endif.endif.endif
  call void @PyErr_SetString(i8* nonnull @PyExc_SystemError, i8* getelementptr inbounds ([43 x i8], [43 x i8]* @".const.unknown error when calling native function", i64 0, i64 0))
  br label %common.ret
}

declare i32 @PyArg_UnpackTuple(i8*, i8*, i64, i64, ...) local_unnamed_addr

declare void @PyErr_SetString(i8*, i8*) local_unnamed_addr

declare i8* @PyNumber_Long(i8*) local_unnamed_addr

declare i64 @PyLong_AsLongLong(i8*) local_unnamed_addr

declare void @Py_DecRef(i8*) local_unnamed_addr

declare i8* @PyErr_Occurred() local_unnamed_addr

declare void @Py_IncRef(i8*) local_unnamed_addr

declare i8* @PyTuple_New(i64) local_unnamed_addr

declare i8* @PyFloat_FromDouble(double) local_unnamed_addr

declare i32 @PyTuple_SetItem(i8*, i64, i8*) local_unnamed_addr

declare void @PyErr_Clear() local_unnamed_addr

declare i8* @PyBytes_FromStringAndSize(i8*, i64) local_unnamed_addr

declare i8* @numba_unpickle(i8*, i32, i8*) local_unnamed_addr

declare i8* @numba_runtime_build_excinfo_struct(i8*, i8*) local_unnamed_addr

declare void @NRT_Free(i8*) local_unnamed_addr

declare void @numba_do_raise(i8*) local_unnamed_addr

declare void @PyErr_SetNone(i8*) local_unnamed_addr

define [5 x double] @cfunc._ZN8__main__5func4B2v9B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx(i64 %.1) local_unnamed_addr {
entry:
  %.3 = alloca [5 x double], align 8
  %.fca.0.gep1 = bitcast [5 x double]* %.3 to double*
  %.fca.1.gep = getelementptr inbounds [5 x double], [5 x double]* %.3, i64 0, i64 1
  %.fca.2.gep = getelementptr inbounds [5 x double], [5 x double]* %.3, i64 0, i64 2
  %.fca.3.gep = getelementptr inbounds [5 x double], [5 x double]* %.3, i64 0, i64 3
  %.fca.4.gep = getelementptr inbounds [5 x double], [5 x double]* %.3, i64 0, i64 4
  %excinfo = alloca { i8*, i32, i8*, i8*, i32 }*, align 8
  %0 = bitcast [5 x double]* %.3 to i8*
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(40) %0, i8 0, i64 40, i1 false)
  store { i8*, i32, i8*, i8*, i32 }* null, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  %.7 = call i32 @_ZN8__main__5func4B2v9B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx([5 x double]* nonnull %.3, { i8*, i32, i8*, i8*, i32 }** nonnull %excinfo, i64 %.1) #2
  %.8 = load { i8*, i32, i8*, i8*, i32 }*, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  %.15 = icmp sgt i32 %.7, 0
  %.16 = select i1 %.15, { i8*, i32, i8*, i8*, i32 }* %.8, { i8*, i32, i8*, i8*, i32 }* undef
  %.17.fca.0.load = load double, double* %.fca.0.gep1, align 8
  %.17.fca.0.insert = insertvalue [5 x double] poison, double %.17.fca.0.load, 0
  %.17.fca.1.load = load double, double* %.fca.1.gep, align 8
  %.17.fca.1.insert = insertvalue [5 x double] %.17.fca.0.insert, double %.17.fca.1.load, 1
  %.17.fca.2.load = load double, double* %.fca.2.gep, align 8
  %.17.fca.2.insert = insertvalue [5 x double] %.17.fca.1.insert, double %.17.fca.2.load, 2
  %.17.fca.3.load = load double, double* %.fca.3.gep, align 8
  %.17.fca.3.insert = insertvalue [5 x double] %.17.fca.2.insert, double %.17.fca.3.load, 3
  %.17.fca.4.load = load double, double* %.fca.4.gep, align 8
  %.17.fca.4.insert = insertvalue [5 x double] %.17.fca.3.insert, double %.17.fca.4.load, 4
  %.19 = alloca i32, align 4
  store i32 0, i32* %.19, align 4
  switch i32 %.7, label %entry.if [
    i32 -2, label %common.ret
    i32 0, label %common.ret
  ]

entry.if:                                         ; preds = %entry
  %1 = icmp sgt i32 %.7, 0
  call void @numba_gil_ensure(i32* nonnull %.19)
  br i1 %1, label %entry.if.if, label %entry.if.endif

common.ret:                                       ; preds = %entry, %entry, %.22, %entry.if.if.if.if
  %common.ret.op = phi [5 x double] [ zeroinitializer, %entry.if.if.if.if ], [ %.17.fca.4.insert, %.22 ], [ %.17.fca.4.insert, %entry ], [ %.17.fca.4.insert, %entry ]
  ret [5 x double] %common.ret.op

.22:                                              ; preds = %entry.if.endif, %entry.if.if.endif, %entry.if.if.endif.if, %entry.if.endif.endif.endif, %entry.if.endif.if
  %.70 = call i8* @PyUnicode_FromString(i8* getelementptr inbounds ([50 x i8], [50 x i8]* @".const.<numba.core.cpu.CPUContext object at 0x110f389b0>", i64 0, i64 0))
  call void @PyErr_WriteUnraisable(i8* %.70)
  call void @Py_DecRef(i8* %.70)
  call void @numba_gil_release(i32* nonnull %.19)
  br label %common.ret

entry.if.if:                                      ; preds = %entry.if
  call void @PyErr_Clear()
  %.25 = load { i8*, i32, i8*, i8*, i32 }, { i8*, i32, i8*, i8*, i32 }* %.16, align 8
  %.26 = extractvalue { i8*, i32, i8*, i8*, i32 } %.25, 4
  %.27 = icmp sgt i32 %.26, 0
  %.30 = extractvalue { i8*, i32, i8*, i8*, i32 } %.25, 0
  %.32 = extractvalue { i8*, i32, i8*, i8*, i32 } %.25, 1
  br i1 %.27, label %entry.if.if.if, label %entry.if.if.else

entry.if.endif:                                   ; preds = %entry.if
  switch i32 %.7, label %entry.if.endif.endif.endif [
    i32 -3, label %entry.if.endif.if
    i32 -1, label %.22
  ]

entry.if.if.if:                                   ; preds = %entry.if.if
  %.33 = sext i32 %.32 to i64
  %.34 = call i8* @PyBytes_FromStringAndSize(i8* %.30, i64 %.33)
  %.35 = load { i8*, i32, i8*, i8*, i32 }, { i8*, i32, i8*, i8*, i32 }* %.16, align 8
  %.36 = extractvalue { i8*, i32, i8*, i8*, i32 } %.35, 2
  %.38 = extractvalue { i8*, i32, i8*, i8*, i32 } %.35, 3
  %.39 = bitcast i8* %.38 to i8* (i8*)*
  %.40 = call i8* %.39(i8* %.36)
  %.41 = icmp eq i8* %.40, null
  br i1 %.41, label %entry.if.if.if.if, label %entry.if.if.if.endif, !prof !0

entry.if.if.else:                                 ; preds = %entry.if.if
  %.54 = extractvalue { i8*, i32, i8*, i8*, i32 } %.25, 2
  %.55 = call i8* @numba_unpickle(i8* %.30, i32 %.32, i8* %.54)
  br label %entry.if.if.endif

entry.if.if.endif:                                ; preds = %entry.if.if.if.endif, %entry.if.if.else
  %.57 = phi i8* [ %.45, %entry.if.if.if.endif ], [ %.55, %entry.if.if.else ]
  %.58.not = icmp eq i8* %.57, null
  br i1 %.58.not, label %.22, label %entry.if.if.endif.if, !prof !0

entry.if.if.if.if:                                ; preds = %entry.if.if.if
  call void @PyErr_SetString(i8* nonnull @PyExc_RuntimeError, i8* getelementptr inbounds ([61 x i8], [61 x i8]* @".const.Error creating Python tuple from runtime exception arguments.1", i64 0, i64 0))
  br label %common.ret

entry.if.if.if.endif:                             ; preds = %entry.if.if.if
  %.45 = call i8* @numba_runtime_build_excinfo_struct(i8* %.34, i8* nonnull %.40)
  %.46 = bitcast { i8*, i32, i8*, i8*, i32 }* %.16 to i8*
  call void @NRT_Free(i8* nonnull %.46)
  br label %entry.if.if.endif

entry.if.if.endif.if:                             ; preds = %entry.if.if.endif
  call void @numba_do_raise(i8* nonnull %.57)
  br label %.22

entry.if.endif.if:                                ; preds = %entry.if.endif
  call void @PyErr_SetNone(i8* nonnull @PyExc_StopIteration)
  br label %.22

entry.if.endif.endif.endif:                       ; preds = %entry.if.endif
  call void @PyErr_SetString(i8* nonnull @PyExc_SystemError, i8* getelementptr inbounds ([43 x i8], [43 x i8]* @".const.unknown error when calling native function.2", i64 0, i64 0))
  br label %.22
}

declare void @numba_gil_ensure(i32*) local_unnamed_addr

declare i8* @PyUnicode_FromString(i8*) local_unnamed_addr

declare void @PyErr_WriteUnraisable(i8*) local_unnamed_addr

declare void @numba_gil_release(i32*) local_unnamed_addr

define linkonce_odr i32 @_ZN8__main__5func3B3v10B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx({ double, double, i64, double, double }* noalias nocapture writeonly %retptr, { i8*, i32, i8*, i8*, i32 }** noalias nocapture writeonly %excinfo, i64 %arg.m) local_unnamed_addr {
B0.endif:
  %.151 = alloca { double, double, i64, i64, double }, align 8
  %excinfo.1 = alloca { i8*, i32, i8*, i8*, i32 }*, align 8
  store { i8*, i32, i8*, i8*, i32 }* null, { i8*, i32, i8*, i8*, i32 }** %excinfo.1, align 8
  %.660 = alloca { double, double, double, i64, i64 }, align 8
  %excinfo.3 = alloca { i8*, i32, i8*, i8*, i32 }*, align 8
  store { i8*, i32, i8*, i8*, i32 }* null, { i8*, i32, i8*, i8*, i32 }** %excinfo.3, align 8
  %.872 = alloca { double, double, double, i64, i64 }, align 8
  %excinfo.5 = alloca { i8*, i32, i8*, i8*, i32 }*, align 8
  store { i8*, i32, i8*, i8*, i32 }* null, { i8*, i32, i8*, i8*, i32 }** %excinfo.5, align 8
  %0 = tail call i64 @llvm.smax.i64(i64 %arg.m, i64 0)
  %.10384.not = icmp slt i64 %arg.m, 1
  br i1 %.10384.not, label %B228.else.if, label %B54.preheader

B54.preheader:                                    ; preds = %B0.endif
  br label %B54

B54:                                              ; preds = %B54.preheader, %B192.else.if
  %lsr.iv26 = phi i64 [ -2, %B54.preheader ], [ %lsr.iv.next27, %B192.else.if ]
  %lsr.iv125 = phi i64 [ %lsr.iv.next126, %B192.else.if ], [ 10, %B54.preheader ]
  %lsr.iv123 = phi i64 [ %13, %B192.else.if ], [ 0, %B54.preheader ]
  %total_sum.4.090 = phi double [ %total_sum.5.1, %B192.else.if ], [ 0.000000e+00, %B54.preheader ]
  %max_value.3.089 = phi double [ %max_value.4.1, %B192.else.if ], [ -1.000000e+07, %B54.preheader ]
  %factorial_sum.3.088 = phi i64 [ %factorial_sum.4.1, %B192.else.if ], [ 0, %B54.preheader ]
  %count_total.3.087 = phi double [ %count_total.4.1, %B192.else.if ], [ 0.000000e+00, %B54.preheader ]
  %.45.085 = phi i64 [ %.116, %B192.else.if ], [ 1, %B54.preheader ]
  %1 = bitcast { double, double, i64, i64, double }* %.151 to i8*
  %2 = add i64 %lsr.iv123, 1
  %3 = udiv i64 %.45.085, 5
  %4 = mul nuw nsw i64 %3, 5
  %5 = add i64 %lsr.iv26, %4
  %6 = udiv i64 %.45.085, 7
  %7 = mul nuw nsw i64 %6, 7
  %8 = add i64 %lsr.iv26, %7
  %9 = udiv i64 %.45.085, 11
  %10 = mul nuw nsw i64 %9, 11
  %11 = add i64 %lsr.iv26, %10
  %12 = add i64 %2, %11
  %13 = add nuw nsw i64 %lsr.iv123, 1
  %14 = add i64 %lsr.iv123, 1
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(40) %1, i8 0, i64 40, i1 false)
  %.155 = call i32 @_ZN8__main__5func2B3v11B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx({ double, double, i64, i64, double }* nonnull %.151, { i8*, i32, i8*, i8*, i32 }** nonnull %excinfo.1, i64 %lsr.iv125)
  switch i32 %.155, label %B54.if [
    i32 -2, label %B54.endif
    i32 0, label %B54.endif
  ]

B270:                                             ; preds = %B270.preheader, %B270.endif
  %lsr.iv121 = phi i64 [ %lsr.iv.next122, %B270.endif ], [ 5, %B270.preheader ]
  %lsr.iv119 = phi i64 [ %lsr.iv.next120, %B270.endif ], [ %spec.select66, %B270.preheader ]
  %max_value.5.082 = phi double [ %.700, %B270.endif ], [ %max_value.3.0.lcssa, %B270.preheader ]
  %total_sum.6.081 = phi double [ %.695, %B270.endif ], [ %total_sum.4.0.lcssa, %B270.preheader ]
  %15 = bitcast { double, double, double, i64, i64 }* %.660 to i8*
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(40) %15, i8 0, i64 40, i1 false)
  %.664 = call i32 @_ZN8__main__5func1B2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx({ double, double, double, i64, i64 }* nonnull %.660, { i8*, i32, i8*, i8*, i32 }** nonnull %excinfo.3, i64 %lsr.iv121)
  switch i32 %.664, label %B270.if [
    i32 -2, label %B270.endif
    i32 0, label %B270.endif
  ]

B348:                                             ; preds = %B270.endif, %B228.else.if
  %total_sum.6.0.lcssa = phi double [ %total_sum.4.0.lcssa, %B228.else.if ], [ %.695, %B270.endif ]
  %max_value.5.0.lcssa = phi double [ %max_value.3.0.lcssa, %B228.else.if ], [ %.700, %B270.endif ]
  %.15.i.i = icmp slt i64 %arg.m, 0
  br i1 %.15.i.i, label %B348.if, label %B0.endif.endif.i.i, !prof !0

B0.endif.endif.i.i:                               ; preds = %B348
  %.29.i.i = tail call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %arg.m, i64 8)
  %.31.i.i = extractvalue { i64, i1 } %.29.i.i, 1
  br i1 %.31.i.i, label %B348.if, label %B0.endif.endif.endif.i.i, !prof !0

B0.endif.endif.endif.i.i:                         ; preds = %B0.endif.endif.i.i
  %.30.i.i = extractvalue { i64, i1 } %.29.i.i, 0
  %.7.i.i.i.i = tail call i8* @NRT_MemInfo_alloc_aligned(i64 %.30.i.i, i32 32), !noalias !23
  %.8.i.i.i.i = icmp eq i8* %.7.i.i.i.i, null
  br i1 %.8.i.i.i.i, label %B348.if, label %B348.endif.endif, !prof !0

B418:                                             ; preds = %B418.preheader, %B418.endif
  %.776.076 = phi i64 [ %17, %B418.endif ], [ 0, %B418.preheader ]
  %16 = bitcast { double, double, double, i64, i64 }* %.872 to i8*
  %17 = add nuw nsw i64 %.776.076, 1
  %18 = add i64 %.776.076, 1
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(40) %16, i8 0, i64 40, i1 false)
  %.876 = call i32 @_ZN8__main__5func1B2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx({ double, double, double, i64, i64 }* nonnull %.872, { i8*, i32, i8*, i8*, i32 }** nonnull %excinfo.5, i64 %18)
  switch i32 %.876, label %B418.if [
    i32 -2, label %B418.endif
    i32 0, label %B418.endif
  ]

B52.endif.i.preheader:                            ; preds = %B418.endif
  %smax = tail call i64 @llvm.smax.i64(i64 %arg.m, i64 1)
  br label %B52.endif.i

B52.endif.i:                                      ; preds = %B52.endif.i, %B52.endif.i.preheader
  %lsr.iv117 = phi double* [ %.6.i1.i.i, %B52.endif.i.preheader ], [ %scevgep, %B52.endif.i ]
  %lsr.iv = phi i64 [ %smax, %B52.endif.i.preheader ], [ %lsr.iv.next, %B52.endif.i ]
  %c.2.06.i = phi double [ 0.000000e+00, %B52.endif.i.preheader ], [ %.239.i, %B52.endif.i ]
  %.236.i = load double, double* %lsr.iv117, align 8, !noalias !36
  %.239.i = fadd double %c.2.06.i, %.236.i
  %lsr.iv.next = add nsw i64 %lsr.iv, -1
  %scevgep = getelementptr double, double* %lsr.iv117, i64 1
  %exitcond.not = icmp eq i64 %lsr.iv.next, 0
  br i1 %exitcond.not, label %B472.endif, label %B52.endif.i, !prof !0

common.ret:                                       ; preds = %B472.endif, %B418.if, %B348.if, %B270.if, %B54.if
  %common.ret.op = phi i32 [ 0, %B472.endif ], [ 1, %B348.if ], [ %.876, %B418.if ], [ %.664, %B270.if ], [ %.155, %B54.if ]
  ret i32 %common.ret.op

B54.if:                                           ; preds = %B54
  %.163 = icmp sgt i32 %.155, 0
  %.156 = load { i8*, i32, i8*, i8*, i32 }*, { i8*, i32, i8*, i8*, i32 }** %excinfo.1, align 8
  %.164 = select i1 %.163, { i8*, i32, i8*, i8*, i32 }* %.156, { i8*, i32, i8*, i8*, i32 }* undef
  store { i8*, i32, i8*, i8*, i32 }* %.164, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  br label %common.ret

B54.endif:                                        ; preds = %B54, %B54
  %19 = bitcast { double, double, i64, i64, double }* %.151 to double*
  %20 = add i64 %2, %5
  %21 = add i64 %2, %8
  %22 = bitcast { double, double, i64, i64, double }* %.151 to i8*
  %sunkaddr28 = getelementptr inbounds i8, i8* %22, i64 32
  %23 = bitcast i8* %sunkaddr28 to double*
  %.165.fca.4.load = load double, double* %23, align 8
  %24 = bitcast { double, double, i64, i64, double }* %.151 to i8*
  %sunkaddr = getelementptr inbounds i8, i8* %24, i64 24
  %25 = bitcast i8* %sunkaddr to i64*
  %.165.fca.3.load = load i64, i64* %25, align 8
  %26 = bitcast { double, double, i64, i64, double }* %.151 to i8*
  %sunkaddr29 = getelementptr inbounds i8, i8* %26, i64 8
  %27 = bitcast i8* %sunkaddr29 to double*
  %.165.fca.1.load = load double, double* %27, align 8
  %.165.fca.0.load = load double, double* %19, align 8
  %.195 = fadd double %total_sum.4.090, %.165.fca.0.load
  %.203 = fcmp ogt double %.165.fca.1.load, %max_value.3.089
  %max_value.4.1 = select i1 %.203, double %.165.fca.1.load, double %max_value.3.089
  %.221 = add nsw i64 %.165.fca.3.load, %factorial_sum.3.088
  %.231 = fadd double %count_total.3.087, %.165.fca.4.load
  %.282 = icmp eq i64 %lsr.iv123, %20
  %.285 = fmul double %.195, 0x3FEE666666666666
  %total_sum.5.1 = select i1 %.282, double %.285, double %.195
  %.340 = icmp eq i64 %lsr.iv123, %21
  br i1 %.340, label %B182.else.if, label %B192.else.if

B182.else.if:                                     ; preds = %B54.endif
  %.363 = sdiv i64 %.221, 2
  %28 = and i64 %.221, -9223372036854775807
  %.366 = icmp eq i64 %28, -9223372036854775807
  %.373 = sext i1 %.366 to i64
  %spec.select64 = add nsw i64 %.363, %.373
  br label %B192.else.if

B192.else.if:                                     ; preds = %B182.else.if, %B54.endif
  %factorial_sum.4.1 = phi i64 [ %.221, %B54.endif ], [ %spec.select64, %B182.else.if ]
  %.435 = icmp eq i64 %lsr.iv123, %12
  %.191.i = mul nsw i64 %14, %14
  %.474 = sitofp i64 %.191.i to double
  %.475 = select i1 %.435, double %.474, double -0.000000e+00
  %count_total.4.1 = fadd double %.475, %.231
  %lsr.iv.next126 = add i64 %lsr.iv125, 10
  %.116 = add nuw i64 %.45.085, 1
  %lsr.iv.next27 = add i64 %lsr.iv26, -1
  %exitcond100.not = icmp eq i64 %0, %13
  br i1 %exitcond100.not, label %B228.else.if, label %B54

B228.else.if:                                     ; preds = %B192.else.if, %B0.endif
  %count_total.3.0.lcssa = phi double [ 0.000000e+00, %B0.endif ], [ %count_total.4.1, %B192.else.if ]
  %factorial_sum.3.0.lcssa = phi i64 [ 0, %B0.endif ], [ %factorial_sum.4.1, %B192.else.if ]
  %max_value.3.0.lcssa = phi double [ -1.000000e+07, %B0.endif ], [ %max_value.4.1, %B192.else.if ]
  %total_sum.4.0.lcssa = phi double [ 0.000000e+00, %B0.endif ], [ %total_sum.5.1, %B192.else.if ]
  %.518 = sdiv i64 %arg.m, 2
  %29 = and i64 %arg.m, -9223372036854775807
  %.521 = icmp eq i64 %29, -9223372036854775807
  %.528 = sext i1 %.521 to i64
  %spec.select66 = add nsw i64 %.518, %.528
  %.62078.not = icmp slt i64 %spec.select66, 1
  br i1 %.62078.not, label %B348, label %B270.preheader

B270.preheader:                                   ; preds = %B228.else.if
  br label %B270

B270.if:                                          ; preds = %B270
  %.672 = icmp sgt i32 %.664, 0
  %.665 = load { i8*, i32, i8*, i8*, i32 }*, { i8*, i32, i8*, i8*, i32 }** %excinfo.3, align 8
  %.673 = select i1 %.672, { i8*, i32, i8*, i8*, i32 }* %.665, { i8*, i32, i8*, i8*, i32 }* undef
  store { i8*, i32, i8*, i8*, i32 }* %.673, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  br label %common.ret

B270.endif:                                       ; preds = %B270, %B270
  %30 = bitcast { double, double, double, i64, i64 }* %.660 to double*
  %31 = bitcast { double, double, double, i64, i64 }* %.660 to i8*
  %sunkaddr30 = getelementptr inbounds i8, i8* %31, i64 8
  %32 = bitcast i8* %sunkaddr30 to double*
  %.674.fca.1.load = load double, double* %32, align 8
  %.674.fca.0.load = load double, double* %30, align 8
  %.695 = fadd double %total_sum.6.081, %.674.fca.0.load
  %.699 = fcmp ogt double %.674.fca.1.load, %max_value.5.082
  %.700 = select i1 %.699, double %.674.fca.1.load, double %max_value.5.082
  %lsr.iv.next120 = add i64 %lsr.iv119, -1
  %lsr.iv.next122 = add i64 %lsr.iv121, 5
  %exitcond99.not = icmp eq i64 %lsr.iv.next120, 0
  br i1 %exitcond99.not, label %B348, label %B270

B348.if:                                          ; preds = %B0.endif.endif.endif.i.i, %B0.endif.endif.i.i, %B348
  %excinfo.1.0.ph.i = phi { i8*, i32, i8*, i8*, i32 }* [ @.const.picklebuf.4575216064.13, %B0.endif.endif.endif.i.i ], [ @.const.picklebuf.4575674432.14, %B0.endif.endif.i.i ], [ @.const.picklebuf.4575674496.15, %B348 ]
  store { i8*, i32, i8*, i8*, i32 }* %excinfo.1.0.ph.i, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  br label %common.ret

B348.endif.endif:                                 ; preds = %B0.endif.endif.endif.i.i
  %33 = icmp slt i64 %arg.m, 1
  %.5.i.i.i = getelementptr i8, i8* %.7.i.i.i.i, i64 24
  %34 = bitcast i8* %.5.i.i.i to double**
  %.6.i1.i.i = load double*, double** %34, align 8, !noalias !39
  %.26.i.i = shl nuw nsw i64 %arg.m, 3
  %.27.i.i = bitcast double* %.6.i1.i.i to i8*
  tail call void @llvm.memset.p0i8.i64(i8* align 1 %.27.i.i, i8 0, i64 %.26.i.i, i1 false), !noalias !40
  br i1 %33, label %B472.endif, label %B418.preheader

B418.preheader:                                   ; preds = %B348.endif.endif
  br label %B418

B418.if:                                          ; preds = %B418
  %.884 = icmp sgt i32 %.876, 0
  %.877 = load { i8*, i32, i8*, i8*, i32 }*, { i8*, i32, i8*, i8*, i32 }** %excinfo.5, align 8
  %.885 = select i1 %.884, { i8*, i32, i8*, i8*, i32 }* %.877, { i8*, i32, i8*, i8*, i32 }* undef
  store { i8*, i32, i8*, i8*, i32 }* %.885, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  br label %common.ret

B418.endif:                                       ; preds = %B418, %B418
  %35 = bitcast { double, double, double, i64, i64 }* %.872 to double*
  %.886.fca.0.load = load double, double* %35, align 8
  %scevgep25 = getelementptr double, double* %.6.i1.i.i, i64 %.776.076
  store double %.886.fca.0.load, double* %scevgep25, align 8
  %exitcond98.not = icmp eq i64 %0, %17
  br i1 %exitcond98.not, label %B52.endif.i.preheader, label %B418

B472.endif:                                       ; preds = %B52.endif.i, %B348.endif.endif
  %c.2.0.lcssa.i = phi double [ 0.000000e+00, %B348.endif.endif ], [ %.239.i, %B52.endif.i ]
  tail call void @NRT_decref(i8* nonnull %.7.i.i.i.i)
  %retptr.repack13331 = bitcast { double, double, i64, double, double }* %retptr to double*
  store double %total_sum.6.0.lcssa, double* %retptr.repack13331, align 8
  %retptr.repack28 = getelementptr inbounds { double, double, i64, double, double }, { double, double, i64, double, double }* %retptr, i64 0, i32 1
  store double %max_value.5.0.lcssa, double* %retptr.repack28, align 8
  %retptr.repack30 = getelementptr inbounds { double, double, i64, double, double }, { double, double, i64, double, double }* %retptr, i64 0, i32 2
  store i64 %factorial_sum.3.0.lcssa, i64* %retptr.repack30, align 8
  %retptr.repack32 = getelementptr inbounds { double, double, i64, double, double }, { double, double, i64, double, double }* %retptr, i64 0, i32 3
  store double %count_total.3.0.lcssa, double* %retptr.repack32, align 8
  %retptr.repack34 = getelementptr inbounds { double, double, i64, double, double }, { double, double, i64, double, double }* %retptr, i64 0, i32 4
  store double %c.2.0.lcssa.i, double* %retptr.repack34, align 8
  br label %common.ret
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare i64 @llvm.smax.i64(i64, i64) #0

; Function Attrs: argmemonly mustprogress nocallback nofree nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #1

define linkonce_odr i32 @_ZN8__main__5func2B3v11B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx({ double, double, i64, i64, double }* noalias nocapture writeonly %retptr, { i8*, i32, i8*, i8*, i32 }** noalias nocapture writeonly %excinfo, i64 %arg.m) local_unnamed_addr {
B0.endif:
  %.150 = alloca { double, double, double, i64, i64 }, align 8
  %excinfo.1 = alloca { i8*, i32, i8*, i8*, i32 }*, align 8
  store { i8*, i32, i8*, i8*, i32 }* null, { i8*, i32, i8*, i8*, i32 }** %excinfo.1, align 8
  %.658 = alloca { double, double, double, i64, i64 }, align 8
  %excinfo.3 = alloca { i8*, i32, i8*, i8*, i32 }*, align 8
  store { i8*, i32, i8*, i8*, i32 }* null, { i8*, i32, i8*, i8*, i32 }** %excinfo.3, align 8
  %.870 = alloca { double, double, double, i64, i64 }, align 8
  %excinfo.5 = alloca { i8*, i32, i8*, i8*, i32 }*, align 8
  store { i8*, i32, i8*, i8*, i32 }* null, { i8*, i32, i8*, i8*, i32 }** %excinfo.5, align 8
  %0 = tail call i64 @llvm.smax.i64(i64 %arg.m, i64 0)
  %.10284.not = icmp slt i64 %arg.m, 1
  br i1 %.10284.not, label %B228.else.if, label %B54.preheader

B54.preheader:                                    ; preds = %B0.endif
  br label %B54

B54:                                              ; preds = %B54.preheader, %B192.else.if
  %lsr.iv125 = phi i64 [ %lsr.iv.next126, %B192.else.if ], [ 10, %B54.preheader ]
  %lsr.iv123 = phi i64 [ %2, %B192.else.if ], [ 0, %B54.preheader ]
  %total_sum.4.090 = phi double [ %total_sum.5.1, %B192.else.if ], [ 0.000000e+00, %B54.preheader ]
  %max_value.3.089 = phi double [ %max_value.4.1, %B192.else.if ], [ -1.000000e+07, %B54.preheader ]
  %factorial_sum.3.088 = phi i64 [ %factorial_sum.4.1, %B192.else.if ], [ 0, %B54.preheader ]
  %count_total.3.087 = phi i64 [ %count_total.4.1, %B192.else.if ], [ 0, %B54.preheader ]
  %.44.085 = phi i64 [ %.115, %B192.else.if ], [ 1, %B54.preheader ]
  %1 = bitcast { double, double, double, i64, i64 }* %.150 to i8*
  %2 = add nuw nsw i64 %lsr.iv123, 1
  %3 = add i64 %lsr.iv123, 1
  %4 = udiv i64 %.44.085, 11
  %5 = mul nuw nsw i64 %4, 11
  %6 = sub i64 1, %5
  %7 = udiv i64 %.44.085, 7
  %8 = mul nuw nsw i64 %7, 7
  %9 = sub i64 1, %8
  %10 = udiv i64 %.44.085, 5
  %11 = mul nuw nsw i64 %10, 5
  %12 = sub i64 1, %11
  %13 = add i64 %lsr.iv123, %6
  %14 = sub i64 %lsr.iv123, %13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(40) %1, i8 0, i64 40, i1 false)
  %.154 = call i32 @_ZN8__main__5func1B2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx({ double, double, double, i64, i64 }* nonnull %.150, { i8*, i32, i8*, i8*, i32 }** nonnull %excinfo.1, i64 %lsr.iv125)
  switch i32 %.154, label %B54.if [
    i32 -2, label %B54.endif
    i32 0, label %B54.endif
  ]

B270:                                             ; preds = %B270.preheader, %B270.endif
  %lsr.iv121 = phi i64 [ %lsr.iv.next122, %B270.endif ], [ 5, %B270.preheader ]
  %lsr.iv119 = phi i64 [ %lsr.iv.next120, %B270.endif ], [ %spec.select66, %B270.preheader ]
  %max_value.5.082 = phi double [ %.698, %B270.endif ], [ %max_value.3.0.lcssa, %B270.preheader ]
  %total_sum.6.081 = phi double [ %.693, %B270.endif ], [ %total_sum.4.0.lcssa, %B270.preheader ]
  %15 = bitcast { double, double, double, i64, i64 }* %.658 to i8*
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(40) %15, i8 0, i64 40, i1 false)
  %.662 = call i32 @_ZN8__main__5func1B2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx({ double, double, double, i64, i64 }* nonnull %.658, { i8*, i32, i8*, i8*, i32 }** nonnull %excinfo.3, i64 %lsr.iv121)
  switch i32 %.662, label %B270.if [
    i32 -2, label %B270.endif
    i32 0, label %B270.endif
  ]

B348:                                             ; preds = %B270.endif, %B228.else.if
  %total_sum.6.0.lcssa = phi double [ %total_sum.4.0.lcssa, %B228.else.if ], [ %.693, %B270.endif ]
  %max_value.5.0.lcssa = phi double [ %max_value.3.0.lcssa, %B228.else.if ], [ %.698, %B270.endif ]
  %.15.i.i = icmp slt i64 %arg.m, 0
  br i1 %.15.i.i, label %B348.if, label %B0.endif.endif.i.i, !prof !0

B0.endif.endif.i.i:                               ; preds = %B348
  %.29.i.i = tail call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %arg.m, i64 8)
  %.31.i.i = extractvalue { i64, i1 } %.29.i.i, 1
  br i1 %.31.i.i, label %B348.if, label %B0.endif.endif.endif.i.i, !prof !0

B0.endif.endif.endif.i.i:                         ; preds = %B0.endif.endif.i.i
  %.30.i.i = extractvalue { i64, i1 } %.29.i.i, 0
  %.7.i.i.i.i = tail call i8* @NRT_MemInfo_alloc_aligned(i64 %.30.i.i, i32 32), !noalias !43
  %.8.i.i.i.i = icmp eq i8* %.7.i.i.i.i, null
  br i1 %.8.i.i.i.i, label %B348.if, label %B348.endif.endif, !prof !0

B418:                                             ; preds = %B418.preheader, %B418.endif
  %.774.076 = phi i64 [ %17, %B418.endif ], [ 0, %B418.preheader ]
  %16 = bitcast { double, double, double, i64, i64 }* %.870 to i8*
  %17 = add nuw nsw i64 %.774.076, 1
  %18 = add i64 %.774.076, 1
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(40) %16, i8 0, i64 40, i1 false)
  %.874 = call i32 @_ZN8__main__5func1B2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx({ double, double, double, i64, i64 }* nonnull %.870, { i8*, i32, i8*, i8*, i32 }** nonnull %excinfo.5, i64 %18)
  switch i32 %.874, label %B418.if [
    i32 -2, label %B418.endif
    i32 0, label %B418.endif
  ]

B52.endif.i.preheader:                            ; preds = %B418.endif
  %smax = tail call i64 @llvm.smax.i64(i64 %arg.m, i64 1)
  br label %B52.endif.i

B52.endif.i:                                      ; preds = %B52.endif.i, %B52.endif.i.preheader
  %lsr.iv117 = phi double* [ %.6.i1.i.i, %B52.endif.i.preheader ], [ %scevgep, %B52.endif.i ]
  %lsr.iv = phi i64 [ %smax, %B52.endif.i.preheader ], [ %lsr.iv.next, %B52.endif.i ]
  %c.2.06.i = phi double [ 0.000000e+00, %B52.endif.i.preheader ], [ %.239.i, %B52.endif.i ]
  %.236.i = load double, double* %lsr.iv117, align 8, !noalias !56
  %.239.i = fadd double %c.2.06.i, %.236.i
  %lsr.iv.next = add nsw i64 %lsr.iv, -1
  %scevgep = getelementptr double, double* %lsr.iv117, i64 1
  %exitcond.not = icmp eq i64 %lsr.iv.next, 0
  br i1 %exitcond.not, label %B472.endif, label %B52.endif.i, !prof !0

common.ret:                                       ; preds = %B472.endif, %B418.if, %B348.if, %B270.if, %B54.if
  %common.ret.op = phi i32 [ 0, %B472.endif ], [ 1, %B348.if ], [ %.874, %B418.if ], [ %.662, %B270.if ], [ %.154, %B54.if ]
  ret i32 %common.ret.op

B54.if:                                           ; preds = %B54
  %.162 = icmp sgt i32 %.154, 0
  %.155 = load { i8*, i32, i8*, i8*, i32 }*, { i8*, i32, i8*, i8*, i32 }** %excinfo.1, align 8
  %.163 = select i1 %.162, { i8*, i32, i8*, i8*, i32 }* %.155, { i8*, i32, i8*, i8*, i32 }* undef
  store { i8*, i32, i8*, i8*, i32 }* %.163, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  br label %common.ret

B54.endif:                                        ; preds = %B54, %B54
  %19 = bitcast { double, double, double, i64, i64 }* %.150 to double*
  %20 = add i64 %lsr.iv123, %9
  %21 = add i64 %lsr.iv123, %12
  %22 = sub i64 %lsr.iv123, %21
  %23 = sub i64 %lsr.iv123, %20
  %24 = bitcast { double, double, double, i64, i64 }* %.150 to i8*
  %sunkaddr31 = getelementptr inbounds i8, i8* %24, i64 32
  %25 = bitcast i8* %sunkaddr31 to i64*
  %.164.fca.4.load = load i64, i64* %25, align 8
  %26 = bitcast { double, double, double, i64, i64 }* %.150 to i8*
  %sunkaddr32 = getelementptr inbounds i8, i8* %26, i64 24
  %27 = bitcast i8* %sunkaddr32 to i64*
  %.164.fca.3.load = load i64, i64* %27, align 8
  %28 = bitcast { double, double, double, i64, i64 }* %.150 to i8*
  %sunkaddr = getelementptr inbounds i8, i8* %28, i64 8
  %29 = bitcast i8* %sunkaddr to double*
  %.164.fca.1.load = load double, double* %29, align 8
  %.164.fca.0.load = load double, double* %19, align 8
  %.194 = fadd double %total_sum.4.090, %.164.fca.0.load
  %.202 = fcmp ogt double %.164.fca.1.load, %max_value.3.089
  %max_value.4.1 = select i1 %.202, double %.164.fca.1.load, double %max_value.3.089
  %.220 = add nsw i64 %.164.fca.3.load, %factorial_sum.3.088
  %.281 = icmp eq i64 %lsr.iv123, %22
  %.284 = fmul double %.194, 0x3FEE666666666666
  %total_sum.5.1 = select i1 %.281, double %.284, double %.194
  %.339 = icmp eq i64 %lsr.iv123, %23
  br i1 %.339, label %B182.else.if, label %B192.else.if

B182.else.if:                                     ; preds = %B54.endif
  %.362 = sdiv i64 %.220, 2
  %30 = and i64 %.220, -9223372036854775807
  %.365 = icmp eq i64 %30, -9223372036854775807
  %.372 = sext i1 %.365 to i64
  %spec.select64 = add nsw i64 %.362, %.372
  br label %B192.else.if

B192.else.if:                                     ; preds = %B182.else.if, %B54.endif
  %factorial_sum.4.1 = phi i64 [ %.220, %B54.endif ], [ %spec.select64, %B182.else.if ]
  %.434 = icmp eq i64 %lsr.iv123, %14
  %.191.i = mul nsw i64 %3, %3
  %.473 = select i1 %.434, i64 %.191.i, i64 0
  %.230 = add i64 %.473, %count_total.3.087
  %count_total.4.1 = add i64 %.230, %.164.fca.4.load
  %lsr.iv.next126 = add i64 %lsr.iv125, 10
  %.115 = add nuw i64 %.44.085, 1
  %exitcond100.not = icmp eq i64 %0, %2
  br i1 %exitcond100.not, label %B228.else.if, label %B54

B228.else.if:                                     ; preds = %B192.else.if, %B0.endif
  %count_total.3.0.lcssa = phi i64 [ 0, %B0.endif ], [ %count_total.4.1, %B192.else.if ]
  %factorial_sum.3.0.lcssa = phi i64 [ 0, %B0.endif ], [ %factorial_sum.4.1, %B192.else.if ]
  %max_value.3.0.lcssa = phi double [ -1.000000e+07, %B0.endif ], [ %max_value.4.1, %B192.else.if ]
  %total_sum.4.0.lcssa = phi double [ 0.000000e+00, %B0.endif ], [ %total_sum.5.1, %B192.else.if ]
  %.516 = sdiv i64 %arg.m, 2
  %31 = and i64 %arg.m, -9223372036854775807
  %.519 = icmp eq i64 %31, -9223372036854775807
  %.526 = sext i1 %.519 to i64
  %spec.select66 = add nsw i64 %.516, %.526
  %.61878.not = icmp slt i64 %spec.select66, 1
  br i1 %.61878.not, label %B348, label %B270.preheader

B270.preheader:                                   ; preds = %B228.else.if
  br label %B270

B270.if:                                          ; preds = %B270
  %.670 = icmp sgt i32 %.662, 0
  %.663 = load { i8*, i32, i8*, i8*, i32 }*, { i8*, i32, i8*, i8*, i32 }** %excinfo.3, align 8
  %.671 = select i1 %.670, { i8*, i32, i8*, i8*, i32 }* %.663, { i8*, i32, i8*, i8*, i32 }* undef
  store { i8*, i32, i8*, i8*, i32 }* %.671, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  br label %common.ret

B270.endif:                                       ; preds = %B270, %B270
  %32 = bitcast { double, double, double, i64, i64 }* %.658 to double*
  %33 = bitcast { double, double, double, i64, i64 }* %.658 to i8*
  %sunkaddr33 = getelementptr inbounds i8, i8* %33, i64 8
  %34 = bitcast i8* %sunkaddr33 to double*
  %.672.fca.1.load = load double, double* %34, align 8
  %.672.fca.0.load = load double, double* %32, align 8
  %.693 = fadd double %total_sum.6.081, %.672.fca.0.load
  %.697 = fcmp ogt double %.672.fca.1.load, %max_value.5.082
  %.698 = select i1 %.697, double %.672.fca.1.load, double %max_value.5.082
  %lsr.iv.next120 = add i64 %lsr.iv119, -1
  %lsr.iv.next122 = add i64 %lsr.iv121, 5
  %exitcond99.not = icmp eq i64 %lsr.iv.next120, 0
  br i1 %exitcond99.not, label %B348, label %B270

B348.if:                                          ; preds = %B0.endif.endif.endif.i.i, %B0.endif.endif.i.i, %B348
  %excinfo.1.0.ph.i = phi { i8*, i32, i8*, i8*, i32 }* [ @.const.picklebuf.4575216064.11, %B0.endif.endif.endif.i.i ], [ @.const.picklebuf.4575674432.12, %B0.endif.endif.i.i ], [ @.const.picklebuf.4575674496.13, %B348 ]
  store { i8*, i32, i8*, i8*, i32 }* %excinfo.1.0.ph.i, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  br label %common.ret

B348.endif.endif:                                 ; preds = %B0.endif.endif.endif.i.i
  %35 = icmp slt i64 %arg.m, 1
  %.5.i.i.i = getelementptr i8, i8* %.7.i.i.i.i, i64 24
  %36 = bitcast i8* %.5.i.i.i to double**
  %.6.i1.i.i = load double*, double** %36, align 8, !noalias !59
  %.26.i.i = shl nuw nsw i64 %arg.m, 3
  %.27.i.i = bitcast double* %.6.i1.i.i to i8*
  tail call void @llvm.memset.p0i8.i64(i8* align 1 %.27.i.i, i8 0, i64 %.26.i.i, i1 false), !noalias !60
  br i1 %35, label %B472.endif, label %B418.preheader

B418.preheader:                                   ; preds = %B348.endif.endif
  br label %B418

B418.if:                                          ; preds = %B418
  %.882 = icmp sgt i32 %.874, 0
  %.875 = load { i8*, i32, i8*, i8*, i32 }*, { i8*, i32, i8*, i8*, i32 }** %excinfo.5, align 8
  %.883 = select i1 %.882, { i8*, i32, i8*, i8*, i32 }* %.875, { i8*, i32, i8*, i8*, i32 }* undef
  store { i8*, i32, i8*, i8*, i32 }* %.883, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  br label %common.ret

B418.endif:                                       ; preds = %B418, %B418
  %37 = bitcast { double, double, double, i64, i64 }* %.870 to double*
  %.884.fca.0.load = load double, double* %37, align 8
  %scevgep28 = getelementptr double, double* %.6.i1.i.i, i64 %.774.076
  store double %.884.fca.0.load, double* %scevgep28, align 8
  %exitcond98.not = icmp eq i64 %0, %17
  br i1 %exitcond98.not, label %B52.endif.i.preheader, label %B418

B472.endif:                                       ; preds = %B52.endif.i, %B348.endif.endif
  %c.2.0.lcssa.i = phi double [ 0.000000e+00, %B348.endif.endif ], [ %.239.i, %B52.endif.i ]
  tail call void @NRT_decref(i8* nonnull %.7.i.i.i.i)
  %retptr.repack1333134 = bitcast { double, double, i64, i64, double }* %retptr to double*
  store double %total_sum.6.0.lcssa, double* %retptr.repack1333134, align 8
  %retptr.repack28 = getelementptr inbounds { double, double, i64, i64, double }, { double, double, i64, i64, double }* %retptr, i64 0, i32 1
  store double %max_value.5.0.lcssa, double* %retptr.repack28, align 8
  %retptr.repack30 = getelementptr inbounds { double, double, i64, i64, double }, { double, double, i64, i64, double }* %retptr, i64 0, i32 2
  store i64 %factorial_sum.3.0.lcssa, i64* %retptr.repack30, align 8
  %retptr.repack32 = getelementptr inbounds { double, double, i64, i64, double }, { double, double, i64, i64, double }* %retptr, i64 0, i32 3
  store i64 %count_total.3.0.lcssa, i64* %retptr.repack32, align 8
  %retptr.repack34 = getelementptr inbounds { double, double, i64, i64, double }, { double, double, i64, i64, double }* %retptr, i64 0, i32 4
  store double %c.2.0.lcssa.i, double* %retptr.repack34, align 8
  br label %common.ret
}

define linkonce_odr i32 @_ZN8__main__5func1B2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx({ double, double, double, i64, i64 }* noalias nocapture writeonly %retptr, { i8*, i32, i8*, i8*, i32 }** noalias nocapture writeonly %excinfo, i64 %arg.n) local_unnamed_addr {
entry:
  %.15.i.i = icmp slt i64 %arg.n, 0
  br i1 %.15.i.i, label %B0.if, label %B0.endif.endif.i.i, !prof !0

B0.endif.endif.i.i:                               ; preds = %entry
  %.29.i.i = tail call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %arg.n, i64 8)
  %.31.i.i = extractvalue { i64, i1 } %.29.i.i, 1
  br i1 %.31.i.i, label %B0.if, label %B0.endif.endif.endif.i.i, !prof !0

B0.endif.endif.endif.i.i:                         ; preds = %B0.endif.endif.i.i
  %.30.i.i = extractvalue { i64, i1 } %.29.i.i, 0
  %.7.i.i.i.i = tail call i8* @NRT_MemInfo_alloc_aligned(i64 %.30.i.i, i32 32), !noalias !63
  %.8.i.i.i.i = icmp eq i8* %.7.i.i.i.i, null
  br i1 %.8.i.i.i.i, label %B0.if, label %B0.endif.endif, !prof !0

B76:                                              ; preds = %B76.preheader65, %B76
  %.78.0134 = phi i64 [ %.147, %B76 ], [ %.78.0134.ph, %B76.preheader65 ]
  %.147 = add nuw nsw i64 %.78.0134, 1
  %.172 = sitofp i64 %.78.0134 to double
  %.173 = fmul double %.172, 1.100000e+00
  %scevgep91 = getelementptr double, double* %.6.i1.i.i, i64 %.78.0134
  store double %.173, double* %scevgep91, align 8
  %exitcond141.not = icmp eq i64 %arg.n, %.147
  br i1 %exitcond141.not, label %B96.endif, label %B76, !llvm.loop !76

for.end.preheader:                                ; preds = %B128.preheader244.thread, %middle.block149, %B96.endif
  %total.2.0.lcssa = phi double [ 0.000000e+00, %B96.endif ], [ %36, %middle.block149 ], [ %.364, %B128.preheader244.thread ]
  br label %for.end

B352.preheader.thread:                            ; preds = %B0.endif.endif
  %.62819 = load double, double* %.6.i1.i.i, align 8
  br label %B670.sink.split

B356.preheader:                                   ; preds = %B292, %B252.endif
  %max_val.2.0.lcssa = phi double [ %.628, %B252.endif ], [ %max_val.3.1, %B292 ]
  %min.iters.check165 = icmp eq i64 %arg.n, 1
  br i1 %min.iters.check165, label %B356.preheader243, label %vector.ph166

vector.ph166:                                     ; preds = %B356.preheader
  %scevgep274 = getelementptr double, double* %.6.i1.i.i, i64 1
  %0 = and i64 %arg.n, -2
  %1 = lshr i64 %arg.n, 1
  %2 = mul i64 %1, -2
  br label %vector.body171

vector.body171:                                   ; preds = %vector.body171, %vector.ph166
  %lsr.iv = phi i64 [ %lsr.iv.next, %vector.body171 ], [ %2, %vector.ph166 ]
  %lsr.iv275 = phi double* [ %scevgep276, %vector.body171 ], [ %scevgep274, %vector.ph166 ]
  %vec.phi173 = phi double [ %6, %vector.body171 ], [ 0.000000e+00, %vector.ph166 ]
  %scevgep83 = getelementptr double, double* %lsr.iv275, i64 -1
  %3 = load double, double* %scevgep83, align 8
  %4 = load double, double* %lsr.iv275, align 8
  %5 = fadd double %vec.phi173, %3
  %6 = fadd double %5, %4
  %scevgep276 = getelementptr double, double* %lsr.iv275, i64 2
  %lsr.iv.next = add i64 %lsr.iv, 2
  %7 = icmp eq i64 %lsr.iv.next, 0
  br i1 %7, label %middle.block163, label %vector.body171, !llvm.loop !79

middle.block163:                                  ; preds = %vector.body171
  %cmp.n170 = icmp eq i64 %0, %arg.n
  br i1 %cmp.n170, label %B376.endif, label %B356.preheader243

B356.preheader243:                                ; preds = %middle.block163, %B356.preheader
  %running_sum.2.0118.ph = phi double [ %6, %middle.block163 ], [ 0.000000e+00, %B356.preheader ]
  %.865.0116.ph = phi i64 [ %0, %middle.block163 ], [ 0, %B356.preheader ]
  %scevgep = getelementptr double, double* %.6.i1.i.i, i64 %.865.0116.ph
  %8 = sub i64 %arg.n, %.865.0116.ph
  br label %B356

B292:                                             ; preds = %B292.preheader, %B292
  %lsr.iv85 = phi double* [ %scevgep84, %B292.preheader ], [ %scevgep87, %B292 ]
  %lsr.iv283 = phi i64 [ %lsr.iv.next284, %B292 ], [ %.669, %B292.preheader ]
  %max_val.2.0123 = phi double [ %max_val.3.1, %B292 ], [ %.628, %B292.preheader ]
  %.783 = load double, double* %lsr.iv85, align 8
  %.785 = fcmp ogt double %.783, %max_val.2.0123
  %max_val.3.1 = select i1 %.785, double %.783, double %max_val.2.0123
  %lsr.iv.next284 = add i64 %lsr.iv283, -1
  %scevgep87 = getelementptr double, double* %lsr.iv85, i64 1
  %exitcond139.not = icmp eq i64 %lsr.iv.next284, 0
  br i1 %exitcond139.not, label %B356.preheader, label %B292

B356:                                             ; preds = %B356, %B356.preheader243
  %lsr.iv81 = phi i64 [ %lsr.iv.next82, %B356 ], [ %8, %B356.preheader243 ]
  %lsr.iv270 = phi double* [ %scevgep271, %B356 ], [ %scevgep, %B356.preheader243 ]
  %running_sum.2.0118 = phi double [ %.990, %B356 ], [ %running_sum.2.0118.ph, %B356.preheader243 ]
  %.988 = load double, double* %lsr.iv270, align 8
  %.990 = fadd double %running_sum.2.0118, %.988
  %scevgep271 = getelementptr double, double* %lsr.iv270, i64 1
  %lsr.iv.next82 = add i64 %lsr.iv81, -1
  %ov = icmp eq i64 %lsr.iv.next82, 0
  br i1 %ov, label %B376.endif, label %B356, !llvm.loop !80

B404.loopexit:                                    ; preds = %B404.loopexit.preheader242, %B404.loopexit
  %lsr.iv77 = phi i64 [ %49, %B404.loopexit.preheader242 ], [ %lsr.iv.next78, %B404.loopexit ]
  %count.3.0113 = phi i64 [ %count.3.0113.ph, %B404.loopexit.preheader242 ], [ %9, %B404.loopexit ]
  %.1085112 = phi i64 [ %.1085112.ph, %B404.loopexit.preheader242 ], [ %.1095, %B404.loopexit ]
  %.1095 = add nsw i64 %.1085112, -1
  %smax = tail call i64 @llvm.smax.i64(i64 %.1085112, i64 0)
  %9 = add i64 %smax, %count.3.0113
  %lsr.iv.next78 = add i64 %lsr.iv77, -1
  %ov133 = icmp eq i64 %lsr.iv.next78, 0
  br i1 %ov133, label %B456, label %B404.loopexit, !llvm.loop !81

B456:                                             ; preds = %B404.loopexit, %middle.block178
  %.lcssa = phi i64 [ %48, %middle.block178 ], [ %9, %B404.loopexit ]
  %10 = icmp eq i64 %arg.n, 1
  br i1 %10, label %B506.preheader, label %vector.ph202

vector.ph202:                                     ; preds = %B456
  %n.vec204 = and i64 %arg.n, -2
  br label %vector.body208

vector.body208:                                   ; preds = %vector.body208, %vector.ph202
  %index209 = phi i64 [ 0, %vector.ph202 ], [ %induction214, %vector.body208 ]
  %vec.phi210 = phi i64 [ 1, %vector.ph202 ], [ %12, %vector.body208 ]
  %vec.phi211 = phi i64 [ 1, %vector.ph202 ], [ %13, %vector.body208 ]
  %induction214 = add i64 %index209, 2
  %11 = add i64 %index209, 1
  %12 = mul i64 %vec.phi210, %11
  %13 = mul i64 %vec.phi211, %induction214
  %14 = icmp eq i64 %n.vec204, %induction214
  br i1 %14, label %middle.block199, label %vector.body208, !llvm.loop !82

middle.block199:                                  ; preds = %vector.body208
  %ind.end205 = or i64 %arg.n, 1
  %bin.rdx216 = mul i64 %13, %12
  %cmp.n207 = icmp eq i64 %n.vec204, %arg.n
  br i1 %cmp.n207, label %B522, label %B506.preheader

B506.preheader:                                   ; preds = %middle.block199, %B456
  %factorial.2.0105.ph = phi i64 [ %bin.rdx216, %middle.block199 ], [ 1, %B456 ]
  %.1310.0103.ph = phi i64 [ %ind.end205, %middle.block199 ], [ 1, %B456 ]
  %15 = add i64 %arg.n, 1
  br label %B506

B506:                                             ; preds = %B506.preheader, %B506
  %factorial.2.0105 = phi i64 [ %.1405, %B506 ], [ %factorial.2.0105.ph, %B506.preheader ]
  %.1310.0103 = phi i64 [ %.1379, %B506 ], [ %.1310.0103.ph, %B506.preheader ]
  %.1379 = add i64 %.1310.0103, 1
  %.1405 = mul nsw i64 %.1310.0103, %factorial.2.0105
  %exitcond136.not = icmp eq i64 %15, %.1379
  br i1 %exitcond136.not, label %B522, label %B506, !llvm.loop !83

B522:                                             ; preds = %B506, %middle.block199
  %.1405.lcssa = phi i64 [ %bin.rdx216, %middle.block199 ], [ %.1405, %B506 ]
  %.1422 = icmp sgt i64 %arg.n, 1
  br i1 %.1422, label %B0.endif.endif.endif.i.i8, label %B670

B0.endif.endif.endif.i.i8:                        ; preds = %B522
  %.7.i.i.i.i6 = tail call i8* @NRT_MemInfo_alloc_aligned(i64 %.30.i.i, i32 32), !noalias !84
  %.8.i.i.i.i7 = icmp eq i8* %.7.i.i.i.i6, null
  br i1 %.8.i.i.i.i7, label %B532.if, label %B532.endif.endif, !prof !0

B624:                                             ; preds = %B624.preheader, %B624
  %lsr.iv111 = phi i64 [ %51, %B624.preheader ], [ %lsr.iv.next112, %B624 ]
  %lsr.iv108 = phi double* [ %scevgep107, %B624.preheader ], [ %scevgep109, %B624 ]
  %store_forwarded = phi double [ 1.000000e+00, %B624.preheader ], [ %.1714, %B624 ]
  %scevgep76 = getelementptr double, double* %lsr.iv108, i64 -2
  %.1713 = load double, double* %scevgep76, align 8
  %.1714 = fadd double %store_forwarded, %.1713
  store double %.1714, double* %lsr.iv108, align 8
  %scevgep109 = getelementptr double, double* %lsr.iv108, i64 1
  %lsr.iv.next112 = add i64 %lsr.iv111, -1
  %exitcond.not = icmp eq i64 %lsr.iv.next112, 0
  br i1 %exitcond.not, label %B670.sink.split, label %B624

common.ret:                                       ; preds = %B532.if, %B0.if, %B670
  %common.ret.op = phi i32 [ 0, %B670 ], [ 1, %B0.if ], [ 1, %B532.if ]
  ret i32 %common.ret.op

B670.sink.split:                                  ; preds = %B624, %B532.endif.endif, %B352.preheader.thread
  %total.2.0.lcssa39 = phi double [ %total.2.0.lcssa, %B532.endif.endif ], [ 0.000000e+00, %B352.preheader.thread ], [ %total.2.0.lcssa, %B624 ]
  %max_val.2.0.lcssa24 = phi double [ %max_val.2.0.lcssa, %B532.endif.endif ], [ %.62819, %B352.preheader.thread ], [ %max_val.2.0.lcssa, %B624 ]
  %.7.i.i.i.i.sink = phi i8* [ %.7.i.i.i.i6, %B532.endif.endif ], [ %.7.i.i.i.i, %B352.preheader.thread ], [ %.7.i.i.i.i6, %B624 ]
  %running_sum.2.0.lcssa146.ph = phi double [ %.990.lcssa, %B532.endif.endif ], [ 0.000000e+00, %B352.preheader.thread ], [ %.990.lcssa, %B624 ]
  %count.3.0.lcssa144.ph = phi i64 [ %.lcssa, %B532.endif.endif ], [ 0, %B352.preheader.thread ], [ %.lcssa, %B624 ]
  %factorial.3.055.ph = phi i64 [ %.1405.lcssa, %B532.endif.endif ], [ 1, %B352.preheader.thread ], [ %.1405.lcssa, %B624 ]
  tail call void @NRT_decref(i8* nonnull %.7.i.i.i.i.sink)
  br label %B670

B670:                                             ; preds = %B670.sink.split, %B522
  %total.2.0.lcssa38 = phi double [ %total.2.0.lcssa, %B522 ], [ %total.2.0.lcssa39, %B670.sink.split ]
  %max_val.2.0.lcssa23 = phi double [ %max_val.2.0.lcssa, %B522 ], [ %max_val.2.0.lcssa24, %B670.sink.split ]
  %running_sum.2.0.lcssa146 = phi double [ %.990.lcssa, %B522 ], [ %running_sum.2.0.lcssa146.ph, %B670.sink.split ]
  %count.3.0.lcssa144 = phi i64 [ %.lcssa, %B522 ], [ %count.3.0.lcssa144.ph, %B670.sink.split ]
  %factorial.3.055 = phi i64 [ %.1405.lcssa, %B522 ], [ %factorial.3.055.ph, %B670.sink.split ]
  %retptr.repack3031349395 = bitcast { double, double, double, i64, i64 }* %retptr to double*
  store double %total.2.0.lcssa38, double* %retptr.repack3031349395, align 8
  %retptr.repack30 = getelementptr inbounds { double, double, double, i64, i64 }, { double, double, double, i64, i64 }* %retptr, i64 0, i32 1
  store double %max_val.2.0.lcssa23, double* %retptr.repack30, align 8
  %retptr.repack32 = getelementptr inbounds { double, double, double, i64, i64 }, { double, double, double, i64, i64 }* %retptr, i64 0, i32 2
  store double %running_sum.2.0.lcssa146, double* %retptr.repack32, align 8
  %retptr.repack34 = getelementptr inbounds { double, double, double, i64, i64 }, { double, double, double, i64, i64 }* %retptr, i64 0, i32 3
  store i64 %factorial.3.055, i64* %retptr.repack34, align 8
  %retptr.repack36 = getelementptr inbounds { double, double, double, i64, i64 }, { double, double, double, i64, i64 }* %retptr, i64 0, i32 4
  store i64 %count.3.0.lcssa144, i64* %retptr.repack36, align 8
  br label %common.ret

B0.if:                                            ; preds = %B0.endif.endif.endif.i.i, %B0.endif.endif.i.i, %entry
  %excinfo.1.0.ph.i = phi { i8*, i32, i8*, i8*, i32 }* [ @.const.picklebuf.4575216064, %B0.endif.endif.endif.i.i ], [ @.const.picklebuf.4575674432, %B0.endif.endif.i.i ], [ @.const.picklebuf.4575674496, %entry ]
  store { i8*, i32, i8*, i8*, i32 }* %excinfo.1.0.ph.i, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  br label %common.ret

B0.endif.endif:                                   ; preds = %B0.endif.endif.endif.i.i
  %.5.i.i.i = getelementptr i8, i8* %.7.i.i.i.i, i64 24
  %16 = bitcast i8* %.5.i.i.i to double**
  %.6.i1.i.i = load double*, double** %16, align 8, !noalias !97
  %.26.i.i = shl nuw nsw i64 %arg.n, 3
  %.27.i.i = bitcast double* %.6.i1.i.i to i8*
  tail call void @llvm.memset.p0i8.i64(i8* align 1 %.27.i.i, i8 0, i64 %.26.i.i, i1 false), !noalias !98
  %.134133.not = icmp eq i64 %arg.n, 0
  br i1 %.134133.not, label %B352.preheader.thread, label %B76.preheader

B76.preheader:                                    ; preds = %B0.endif.endif
  %min.iters.check = icmp ult i64 %arg.n, 4
  br i1 %min.iters.check, label %B76.preheader65, label %vector.ph

B76.preheader65:                                  ; preds = %middle.block, %B76.preheader
  %.78.0134.ph = phi i64 [ %17, %middle.block ], [ 0, %B76.preheader ]
  br label %B76

vector.ph:                                        ; preds = %B76.preheader
  %scevgep296 = getelementptr double, double* %.6.i1.i.i, i64 2
  %17 = and i64 %arg.n, -4
  %18 = lshr i64 %arg.n, 2
  %19 = mul i64 %18, -4
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %lsr.iv92 = phi i64 [ %lsr.iv.next93, %vector.body ], [ %19, %vector.ph ]
  %lsr.iv297 = phi double* [ %scevgep298, %vector.body ], [ %scevgep296, %vector.ph ]
  %vec.ind = phi <2 x i64> [ %vec.ind.next, %vector.body ], [ <i64 0, i64 1>, %vector.ph ]
  %lsr.iv297299 = bitcast double* %lsr.iv297 to <2 x double>*
  %step.add = add <2 x i64> %vec.ind, <i64 2, i64 2>
  %20 = sitofp <2 x i64> %vec.ind to <2 x double>
  %21 = sitofp <2 x i64> %step.add to <2 x double>
  %22 = fmul <2 x double> %20, <double 1.100000e+00, double 1.100000e+00>
  %23 = fmul <2 x double> %21, <double 1.100000e+00, double 1.100000e+00>
  %scevgep94 = getelementptr <2 x double>, <2 x double>* %lsr.iv297299, i64 -1
  store <2 x double> %22, <2 x double>* %scevgep94, align 8
  store <2 x double> %23, <2 x double>* %lsr.iv297299, align 8
  %vec.ind.next = add <2 x i64> %vec.ind, <i64 4, i64 4>
  %scevgep298 = getelementptr double, double* %lsr.iv297, i64 4
  %lsr.iv.next93 = add i64 %lsr.iv92, 4
  %24 = icmp eq i64 %lsr.iv.next93, 0
  br i1 %24, label %middle.block, label %vector.body, !llvm.loop !101

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %17, %arg.n
  br i1 %cmp.n, label %B96.endif, label %B76.preheader65

B96.endif:                                        ; preds = %B76, %middle.block
  %25 = and i64 %arg.n, -9223372036854775807
  %.271 = icmp eq i64 %25, 1
  %.2729498 = lshr i64 %arg.n, 1
  %.273 = zext i1 %.271 to i64
  %.274 = add nuw i64 %.2729498, %.273
  %.274.fr = freeze i64 %.274
  switch i64 %.274.fr, label %vector.ph152 [
    i64 0, label %for.end.preheader
    i64 1, label %B128.preheader244.thread
  ]

vector.ph152:                                     ; preds = %B96.endif
  %uglygep2571012 = getelementptr double, double* %.6.i1.i.i, i64 2
  %26 = and i64 %.274.fr, -2
  %27 = lshr i64 %.274.fr, 1
  %28 = mul i64 %27, -2
  br label %vector.body159

vector.body159:                                   ; preds = %vector.body159, %vector.ph152
  %lsr.iv89 = phi i64 [ %lsr.iv.next90, %vector.body159 ], [ %28, %vector.ph152 ]
  %lsr.iv291 = phi i64 [ %lsr.iv.next292, %vector.body159 ], [ 0, %vector.ph152 ]
  %vec.phi = phi double [ %36, %vector.body159 ], [ 0.000000e+00, %vector.ph152 ]
  %29 = icmp slt i64 %lsr.iv291, 0
  %30 = select i1 %29, i64 %arg.n, i64 0
  %31 = add i64 %30, %lsr.iv291
  %32 = getelementptr double, double* %.6.i1.i.i, i64 %31
  %uglygep25811 = getelementptr double, double* %uglygep2571012, i64 %31
  %33 = load double, double* %32, align 8
  %34 = load double, double* %uglygep25811, align 8
  %35 = fadd double %vec.phi, %33
  %36 = fadd double %35, %34
  %lsr.iv.next292 = add i64 %lsr.iv291, 4
  %lsr.iv.next90 = add i64 %lsr.iv89, 2
  %37 = icmp eq i64 %lsr.iv.next90, 0
  br i1 %37, label %middle.block149, label %vector.body159, !llvm.loop !102

middle.block149:                                  ; preds = %vector.body159
  %cmp.n158 = icmp eq i64 %.274.fr, %26
  br i1 %cmp.n158, label %for.end.preheader, label %B128.preheader244

B128.preheader244:                                ; preds = %middle.block149
  %ind.end156 = shl i64 %26, 1
  %.346 = icmp slt i64 %ind.end156, 0
  %spec.select = select i1 %.346, i64 %arg.n, i64 0
  %38 = add i64 %spec.select, %ind.end156
  br label %B128.preheader244.thread

B128.preheader244.thread:                         ; preds = %B128.preheader244, %B96.endif
  %total.2.0131.ph35 = phi double [ %36, %B128.preheader244 ], [ 0.000000e+00, %B96.endif ]
  %.348 = phi i64 [ %38, %B128.preheader244 ], [ 0, %B96.endif ]
  %.361 = getelementptr double, double* %.6.i1.i.i, i64 %.348
  %.362 = load double, double* %.361, align 8
  %.364 = fadd double %total.2.0131.ph35, %.362
  br label %for.end.preheader

for.end:                                          ; preds = %for.end, %for.end.preheader
  %lsr.iv123 = phi i64 [ 0, %for.end.preheader ], [ %lsr.iv.next124, %for.end ]
  %scevgep88 = getelementptr double, double* %.6.i1.i.i, i64 %lsr.iv123
  %.519 = load double, double* %scevgep88, align 8
  %.520 = tail call double @llvm.pow.f64(double %.519, double 1.500000e+00)
  %.558 = tail call double @llvm.sin.f64(double %.519)
  %.563 = fsub double %.520, %.558
  store double %.563, double* %scevgep88, align 8
  %lsr.iv.next124 = add i64 %lsr.iv123, 1
  %exitcond140.not = icmp eq i64 %arg.n, %lsr.iv.next124
  br i1 %exitcond140.not, label %B252.endif, label %for.end

B252.endif:                                       ; preds = %for.end
  %.628 = load double, double* %.6.i1.i.i, align 8
  %.669 = add i64 %arg.n, -1
  %.712120.not = icmp slt i64 %.669, 1
  br i1 %.712120.not, label %B356.preheader, label %B292.preheader

B292.preheader:                                   ; preds = %B252.endif
  %scevgep84 = getelementptr double, double* %.6.i1.i.i, i64 1
  br label %B292

B376.endif:                                       ; preds = %B356, %middle.block163
  %.990.lcssa = phi double [ %6, %middle.block163 ], [ %.990, %B356 ]
  %39 = icmp ult i64 %arg.n, 4
  tail call void @NRT_decref(i8* nonnull %.7.i.i.i.i)
  br i1 %39, label %B404.loopexit.preheader242, label %vector.ph181

vector.ph181:                                     ; preds = %B376.endif
  %.splatinsert = insertelement <2 x i64> poison, i64 %arg.n, i64 0
  %.splat = shufflevector <2 x i64> %.splatinsert, <2 x i64> poison, <2 x i32> zeroinitializer
  %induction192 = add <2 x i64> %.splat, <i64 0, i64 -1>
  %40 = and i64 %arg.n, -4
  %41 = lshr i64 %arg.n, 2
  %42 = mul i64 %41, -4
  br label %vector.body188

vector.body188:                                   ; preds = %vector.body188, %vector.ph181
  %lsr.iv79 = phi i64 [ %lsr.iv.next80, %vector.body188 ], [ %42, %vector.ph181 ]
  %vec.phi190 = phi <2 x i64> [ %45, %vector.body188 ], [ zeroinitializer, %vector.ph181 ]
  %vec.phi191 = phi <2 x i64> [ %46, %vector.body188 ], [ zeroinitializer, %vector.ph181 ]
  %vec.ind193 = phi <2 x i64> [ %vec.ind.next196, %vector.body188 ], [ %induction192, %vector.ph181 ]
  %step.add194 = add <2 x i64> %vec.ind193, <i64 -2, i64 -2>
  %43 = tail call <2 x i64> @llvm.smax.v2i64(<2 x i64> %vec.ind193, <2 x i64> zeroinitializer)
  %44 = tail call <2 x i64> @llvm.smax.v2i64(<2 x i64> %step.add194, <2 x i64> zeroinitializer)
  %45 = add <2 x i64> %43, %vec.phi190
  %46 = add <2 x i64> %44, %vec.phi191
  %vec.ind.next196 = add <2 x i64> %vec.ind193, <i64 -4, i64 -4>
  %lsr.iv.next80 = add i64 %lsr.iv79, 4
  %47 = icmp eq i64 %lsr.iv.next80, 0
  br i1 %47, label %middle.block178, label %vector.body188, !llvm.loop !103

middle.block178:                                  ; preds = %vector.body188
  %ind.end184 = and i64 %arg.n, 3
  %bin.rdx = add <2 x i64> %46, %45
  %48 = tail call i64 @llvm.vector.reduce.add.v2i64(<2 x i64> %bin.rdx)
  %cmp.n187 = icmp eq i64 %40, %arg.n
  br i1 %cmp.n187, label %B456, label %B404.loopexit.preheader242

B404.loopexit.preheader242:                       ; preds = %middle.block178, %B376.endif
  %count.3.0113.ph = phi i64 [ %48, %middle.block178 ], [ 0, %B376.endif ]
  %.1085112.ph = phi i64 [ %ind.end184, %middle.block178 ], [ %arg.n, %B376.endif ]
  %.1030.0111.ph = phi i64 [ %40, %middle.block178 ], [ 0, %B376.endif ]
  %49 = sub i64 %arg.n, %.1030.0111.ph
  br label %B404.loopexit

B532.if:                                          ; preds = %B0.endif.endif.endif.i.i8
  store { i8*, i32, i8*, i8*, i32 }* @.const.picklebuf.4575216064, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  br label %common.ret

B532.endif.endif:                                 ; preds = %B0.endif.endif.endif.i.i8
  %.5.i.i.i12 = getelementptr i8, i8* %.7.i.i.i.i6, i64 24
  %50 = bitcast i8* %.5.i.i.i12 to double**
  %.6.i1.i.i13 = load double*, double** %50, align 8, !noalias !104
  %.27.i.i15 = bitcast double* %.6.i1.i.i13 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* align 1 %.27.i.i15, i8 0, i64 %.26.i.i, i1 false), !noalias !105
  store double 0.000000e+00, double* %.6.i1.i.i13, align 8
  %.1529 = getelementptr double, double* %.6.i1.i.i13, i64 1
  store double 1.000000e+00, double* %.1529, align 8
  %.161499 = icmp ugt i64 %arg.n, 2
  br i1 %.161499, label %B624.preheader, label %B670.sink.split

B624.preheader:                                   ; preds = %B532.endif.endif
  %scevgep107 = getelementptr double, double* %.6.i1.i.i13, i64 2
  %51 = add nsw i64 %arg.n, -2
  br label %B624
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare { i64, i1 } @llvm.smul.with.overflow.i64(i64, i64) #0

declare noalias i8* @NRT_MemInfo_alloc_aligned(i64, i32) local_unnamed_addr

; Function Attrs: noinline
define linkonce_odr void @NRT_decref(i8* %.1) local_unnamed_addr #2 {
.3:
  %.4 = icmp eq i8* %.1, null
  br i1 %.4, label %common.ret, label %.3.endif, !prof !0

common.ret:                                       ; preds = %.3, %.3.endif
  ret void

.3.endif:                                         ; preds = %.3
  fence release
  %.8 = bitcast i8* %.1 to i64*
  %.4.i = atomicrmw sub i64* %.8, i64 1 monotonic, align 8
  %.10 = icmp eq i64 %.4.i, 1
  br i1 %.10, label %.3.endif.if, label %common.ret, !prof !0

.3.endif.if:                                      ; preds = %.3.endif
  fence acquire
  tail call void @NRT_MemInfo_call_dtor(i8* nonnull %.1)
  ret void
}

declare void @NRT_MemInfo_call_dtor(i8*) local_unnamed_addr

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.pow.f64(double, double) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.sin.f64(double) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare <2 x i64> @llvm.smax.v2i64(<2 x i64>, <2 x i64>) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone willreturn
declare i64 @llvm.vector.reduce.add.v2i64(<2 x i64>) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.floor.f64(double) #0

attributes #0 = { mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #1 = { argmemonly mustprogress nocallback nofree nounwind willreturn writeonly }
attributes #2 = { noinline }
attributes #3 = { mustprogress nocallback nofree nosync nounwind readnone willreturn }

!0 = !{!"branch_weights", i32 1, i32 99}
!1 = !{!2, !4, !5, !7, !8, !10, !11, !13}
!2 = distinct !{!2, !3, !"_ZN5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %retptr"}
!3 = distinct !{!3, !"_ZN5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj"}
!4 = distinct !{!4, !3, !"_ZN5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %excinfo"}
!5 = distinct !{!5, !6, !"_ZN5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %retptr"}
!6 = distinct !{!6, !"_ZN5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj"}
!7 = distinct !{!7, !6, !"_ZN5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %excinfo"}
!8 = distinct !{!8, !9, !"_ZN5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29: %retptr"}
!9 = distinct !{!9, !"_ZN5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29"}
!10 = distinct !{!10, !9, !"_ZN5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29: %excinfo"}
!11 = distinct !{!11, !12, !"_ZN5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29: %retptr"}
!12 = distinct !{!12, !"_ZN5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29"}
!13 = distinct !{!13, !12, !"_ZN5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29: %excinfo"}
!14 = !{!15}
!15 = distinct !{!15, !16, !"_ZN5numba2np13old_arraymath9array_sum12_3clocals_3e14array_sum_implB3v13B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAE5ArrayIdLi1E1C7mutable7alignedE: %retptr"}
!16 = distinct !{!16, !"_ZN5numba2np13old_arraymath9array_sum12_3clocals_3e14array_sum_implB3v13B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAE5ArrayIdLi1E1C7mutable7alignedE"}
!17 = !{!"branch_weights", i32 99, i32 101}
!18 = !{!8, !10, !11, !13}
!19 = !{!20, !11, !13}
!20 = distinct !{!20, !21, !"_ZN5numba2np8arrayobj18ol_array_zero_fill12_3clocals_3e4implB2v8B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dE5ArrayIdLi1E1C7mutable7alignedE: %retptr"}
!21 = distinct !{!21, !"_ZN5numba2np8arrayobj18ol_array_zero_fill12_3clocals_3e4implB2v8B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dE5ArrayIdLi1E1C7mutable7alignedE"}
!22 = !{!"branch_weights", i32 99, i32 1}
!23 = !{!24, !26, !27, !29, !30, !32, !33, !35}
!24 = distinct !{!24, !25, !"_ZN5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %retptr"}
!25 = distinct !{!25, !"_ZN5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj"}
!26 = distinct !{!26, !25, !"_ZN5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %excinfo"}
!27 = distinct !{!27, !28, !"_ZN5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %retptr"}
!28 = distinct !{!28, !"_ZN5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj"}
!29 = distinct !{!29, !28, !"_ZN5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %excinfo"}
!30 = distinct !{!30, !31, !"_ZN5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29: %retptr"}
!31 = distinct !{!31, !"_ZN5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29"}
!32 = distinct !{!32, !31, !"_ZN5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29: %excinfo"}
!33 = distinct !{!33, !34, !"_ZN5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29: %retptr"}
!34 = distinct !{!34, !"_ZN5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29"}
!35 = distinct !{!35, !34, !"_ZN5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29: %excinfo"}
!36 = !{!37}
!37 = distinct !{!37, !38, !"_ZN5numba2np13old_arraymath9array_sum12_3clocals_3e14array_sum_implB3v13B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAE5ArrayIdLi1E1C7mutable7alignedE: %retptr"}
!38 = distinct !{!38, !"_ZN5numba2np13old_arraymath9array_sum12_3clocals_3e14array_sum_implB3v13B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAE5ArrayIdLi1E1C7mutable7alignedE"}
!39 = !{!30, !32, !33, !35}
!40 = !{!41, !33, !35}
!41 = distinct !{!41, !42, !"_ZN5numba2np8arrayobj18ol_array_zero_fill12_3clocals_3e4implB2v8B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dE5ArrayIdLi1E1C7mutable7alignedE: %retptr"}
!42 = distinct !{!42, !"_ZN5numba2np8arrayobj18ol_array_zero_fill12_3clocals_3e4implB2v8B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dE5ArrayIdLi1E1C7mutable7alignedE"}
!43 = !{!44, !46, !47, !49, !50, !52, !53, !55}
!44 = distinct !{!44, !45, !"_ZN5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %retptr"}
!45 = distinct !{!45, !"_ZN5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj"}
!46 = distinct !{!46, !45, !"_ZN5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %excinfo"}
!47 = distinct !{!47, !48, !"_ZN5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %retptr"}
!48 = distinct !{!48, !"_ZN5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj"}
!49 = distinct !{!49, !48, !"_ZN5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %excinfo"}
!50 = distinct !{!50, !51, !"_ZN5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29: %retptr"}
!51 = distinct !{!51, !"_ZN5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29"}
!52 = distinct !{!52, !51, !"_ZN5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29: %excinfo"}
!53 = distinct !{!53, !54, !"_ZN5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29: %retptr"}
!54 = distinct !{!54, !"_ZN5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29"}
!55 = distinct !{!55, !54, !"_ZN5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29: %excinfo"}
!56 = !{!57}
!57 = distinct !{!57, !58, !"_ZN5numba2np13old_arraymath9array_sum12_3clocals_3e14array_sum_implB3v13B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAE5ArrayIdLi1E1C7mutable7alignedE: %retptr"}
!58 = distinct !{!58, !"_ZN5numba2np13old_arraymath9array_sum12_3clocals_3e14array_sum_implB3v13B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAE5ArrayIdLi1E1C7mutable7alignedE"}
!59 = !{!50, !52, !53, !55}
!60 = !{!61, !53, !55}
!61 = distinct !{!61, !62, !"_ZN5numba2np8arrayobj18ol_array_zero_fill12_3clocals_3e4implB2v8B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dE5ArrayIdLi1E1C7mutable7alignedE: %retptr"}
!62 = distinct !{!62, !"_ZN5numba2np8arrayobj18ol_array_zero_fill12_3clocals_3e4implB2v8B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dE5ArrayIdLi1E1C7mutable7alignedE"}
!63 = !{!64, !66, !67, !69, !70, !72, !73, !75}
!64 = distinct !{!64, !65, !"_ZN5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %retptr"}
!65 = distinct !{!65, !"_ZN5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj"}
!66 = distinct !{!66, !65, !"_ZN5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %excinfo"}
!67 = distinct !{!67, !68, !"_ZN5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %retptr"}
!68 = distinct !{!68, !"_ZN5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj"}
!69 = distinct !{!69, !68, !"_ZN5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %excinfo"}
!70 = distinct !{!70, !71, !"_ZN5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29: %retptr"}
!71 = distinct !{!71, !"_ZN5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29"}
!72 = distinct !{!72, !71, !"_ZN5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29: %excinfo"}
!73 = distinct !{!73, !74, !"_ZN5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29: %retptr"}
!74 = distinct !{!74, !"_ZN5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29"}
!75 = distinct !{!75, !74, !"_ZN5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29: %excinfo"}
!76 = distinct !{!76, !77, !78}
!77 = !{!"llvm.loop.unroll.runtime.disable"}
!78 = !{!"llvm.loop.isvectorized", i32 1}
!79 = distinct !{!79, !78}
!80 = distinct !{!80, !78}
!81 = distinct !{!81, !77, !78}
!82 = distinct !{!82, !78}
!83 = distinct !{!83, !78}
!84 = !{!85, !87, !88, !90, !91, !93, !94, !96}
!85 = distinct !{!85, !86, !"_ZN5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %retptr"}
!86 = distinct !{!86, !"_ZN5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj"}
!87 = distinct !{!87, !86, !"_ZN5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %excinfo"}
!88 = distinct !{!88, !89, !"_ZN5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %retptr"}
!89 = distinct !{!89, !"_ZN5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj"}
!90 = distinct !{!90, !89, !"_ZN5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %excinfo"}
!91 = distinct !{!91, !92, !"_ZN5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29: %retptr"}
!92 = distinct !{!92, !"_ZN5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29"}
!93 = distinct !{!93, !92, !"_ZN5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29: %excinfo"}
!94 = distinct !{!94, !95, !"_ZN5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29: %retptr"}
!95 = distinct !{!95, !"_ZN5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29"}
!96 = distinct !{!96, !95, !"_ZN5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29: %excinfo"}
!97 = !{!70, !72, !73, !75}
!98 = !{!99, !73, !75}
!99 = distinct !{!99, !100, !"_ZN5numba2np8arrayobj18ol_array_zero_fill12_3clocals_3e4implB2v8B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dE5ArrayIdLi1E1C7mutable7alignedE: %retptr"}
!100 = distinct !{!100, !"_ZN5numba2np8arrayobj18ol_array_zero_fill12_3clocals_3e4implB2v8B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dE5ArrayIdLi1E1C7mutable7alignedE"}
!101 = distinct !{!101, !78}
!102 = distinct !{!102, !78}
!103 = distinct !{!103, !78}
!104 = !{!91, !93, !94, !96}
!105 = !{!106, !94, !96}
!106 = distinct !{!106, !107, !"_ZN5numba2np8arrayobj18ol_array_zero_fill12_3clocals_3e4implB2v8B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dE5ArrayIdLi1E1C7mutable7alignedE: %retptr"}
!107 = distinct !{!107, !"_ZN5numba2np8arrayobj18ol_array_zero_fill12_3clocals_3e4implB2v8B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dE5ArrayIdLi1E1C7mutable7alignedE"}

(venv) mikhailgoykhman@Mikhails-MacBook-Pro-2 core %


"""

"""
; ModuleID = 'func4'
source_filename = "<string>"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin24.3.0"

@.const.func4 = internal constant [6 x i8] c"func4\00"
@_ZN08NumbaEnv8__main__5func4B3v17B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx = common local_unnamed_addr global i8* null
@".const.missing Environment: _ZN08NumbaEnv8__main__5func4B3v17B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx" = internal constant [98 x i8] c"missing Environment: _ZN08NumbaEnv8__main__5func4B3v17B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx\00"
@".const.Error creating Python tuple from runtime exception arguments" = internal constant [61 x i8] c"Error creating Python tuple from runtime exception arguments\00"
@".const.unknown error when calling native function" = internal constant [43 x i8] c"unknown error when calling native function\00"
@".const.Error creating Python tuple from runtime exception arguments.1" = internal constant [61 x i8] c"Error creating Python tuple from runtime exception arguments\00"
@PyExc_SystemError = external global i8
@".const.unknown error when calling native function.2" = internal constant [43 x i8] c"unknown error when calling native function\00"
@".const.<numba.core.cpu.CPUContext object at 0x1089cca10>" = internal constant [50 x i8] c"<numba.core.cpu.CPUContext object at 0x1089cca10>\00"
@_ZN08NumbaEnv8__main__7__func3B3v16B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx = common local_unnamed_addr global i8* null
@_ZN08NumbaEnv5numba7cpython11old_numbers14int_power_impl12_3clocals_3e9int_powerB3v11B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAExx = common local_unnamed_addr global i8* null
@_ZN08NumbaEnv8__main__7__func2B3v14B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx = common local_unnamed_addr global i8* null
@_ZN08NumbaEnv5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29 = common local_unnamed_addr global i8* null
@_ZN08NumbaEnv5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29 = common local_unnamed_addr global i8* null
@.const.pickledata.4433068544 = internal constant [77 x i8] c"\80\04\95B\00\00\00\00\00\00\00\8C\08builtins\94\8C\0AValueError\94\93\94\8C\1Fnegative dimensions not allowed\94\85\94N\87\94."
@.const.pickledata.4433068544.sha1 = internal constant [20 x i8] c"3\1B\85c\BD\B9\DA\C8\1B8B\22s\05,Ho\C1pk"
@.const.picklebuf.4433068544 = internal constant { i8*, i32, i8*, i8*, i32 } { i8* getelementptr inbounds ([77 x i8], [77 x i8]* @.const.pickledata.4433068544, i32 0, i32 0), i32 77, i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.const.pickledata.4433068544.sha1, i32 0, i32 0), i8* null, i32 0 }
@.const.pickledata.4433068480 = internal constant [137 x i8] c"\80\04\95~\00\00\00\00\00\00\00\8C\08builtins\94\8C\0AValueError\94\93\94\8C[array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.\94\85\94N\87\94."
@.const.pickledata.4433068480.sha1 = internal constant [20 x i8] c"X\E1N\CC\B5\07\B1\E0 i\81t\02#\E6\85\CB\8C<W"
@.const.picklebuf.4433068480 = internal constant { i8*, i32, i8*, i8*, i32 } { i8* getelementptr inbounds ([137 x i8], [137 x i8]* @.const.pickledata.4433068480, i32 0, i32 0), i32 137, i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.const.pickledata.4433068480.sha1, i32 0, i32 0), i8* null, i32 0 }
@_ZN08NumbaEnv5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj = common local_unnamed_addr global i8* null
@_ZN08NumbaEnv5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj = common local_unnamed_addr global i8* null
@.const.pickledata.4432610432 = internal constant [86 x i8] c"\80\04\95K\00\00\00\00\00\00\00\8C\08builtins\94\8C\0BMemoryError\94\93\94\8C'Allocation failed (probably too large).\94\85\94N\87\94."
@.const.pickledata.4432610432.sha1 = internal constant [20 x i8] c"\BA(\9D\81\F0\\p \F3G|\15sH\04\DFe\AB\E2\09"
@.const.picklebuf.4432610432 = internal constant { i8*, i32, i8*, i8*, i32 } { i8* getelementptr inbounds ([86 x i8], [86 x i8]* @.const.pickledata.4432610432, i32 0, i32 0), i32 86, i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.const.pickledata.4432610432.sha1, i32 0, i32 0), i8* null, i32 0 }
@_ZN08NumbaEnv5numba2np8arrayobj18ol_array_zero_fill12_3clocals_3e4implB2v8B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dE5ArrayIdLi1E1C7mutable7alignedE = common local_unnamed_addr global i8* null
@_ZN08NumbaEnv8__main__7__func1B2v9B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx = common local_unnamed_addr global i8* null
@PyExc_RuntimeError = external global i8
@_ZN08NumbaEnv5numba2np13old_arraymath9array_sum12_3clocals_3e14array_sum_implB3v12B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAE5ArrayIdLi1E1C7mutable7alignedE = common local_unnamed_addr global i8* null
@_ZN08NumbaEnv5numba2np8arrayobj15make_nditer_cls12_3clocals_3e6NdIter13init_specific12_3clocals_3e11check_shapeB3v13B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAE8UniTupleIxLi1EE8UniTupleIxLi1EE = common local_unnamed_addr global i8* null

define i32 @_ZN8__main__5func4B3v17B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx({ double, double, double, i64, i64 }* noalias nocapture writeonly %retptr, { i8*, i32, i8*, i8*, i32 }** noalias nocapture writeonly %excinfo, i64 %arg.m) local_unnamed_addr {
B0.endif:
  %0 = tail call i64 @llvm.smax.i64(i64 %arg.m, i64 0)
  %.102106.not = icmp slt i64 %arg.m, 1
  br i1 %.102106.not, label %B228.else.if, label %B54.endif.preheader

B54.endif.preheader:                              ; preds = %B0.endif
  br label %B54.endif

B348:                                             ; preds = %B270.endif, %B228.else.if
  %total_sum.6.0.lcssa = phi double [ %total_sum.4.0.lcssa, %B228.else.if ], [ %.693, %B270.endif ]
  %max_value.5.0.lcssa = phi double [ %max_value.3.0.lcssa, %B228.else.if ], [ %.698, %B270.endif ]
  %.15.i.i = icmp slt i64 %arg.m, 0
  br i1 %.15.i.i, label %B348.if, label %B0.endif.endif.i.i, !prof !0

B0.endif.endif.i.i:                               ; preds = %B348
  %.29.i.i = tail call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %arg.m, i64 8)
  %.31.i.i = extractvalue { i64, i1 } %.29.i.i, 1
  br i1 %.31.i.i, label %B348.if, label %B0.endif.endif.endif.i.i, !prof !0

B0.endif.endif.endif.i.i:                         ; preds = %B0.endif.endif.i.i
  %.30.i.i = extractvalue { i64, i1 } %.29.i.i, 0
  %.7.i.i.i.i = tail call i8* @NRT_MemInfo_alloc_aligned(i64 %.30.i.i, i32 32), !noalias !1
  %.8.i.i.i.i = icmp eq i8* %.7.i.i.i.i, null
  br i1 %.8.i.i.i.i, label %B348.if, label %B348.endif.endif, !prof !0

B52.endif.i.preheader:                            ; preds = %B418.endif
  %smax = call i64 @llvm.smax.i64(i64 %arg.m, i64 1)
  br label %B52.endif.i

B52.endif.i:                                      ; preds = %B52.endif.i.preheader, %B52.endif.i
  %lsr.iv121 = phi double* [ %.6.i1.i.i, %B52.endif.i.preheader ], [ %scevgep, %B52.endif.i ]
  %lsr.iv = phi i64 [ %smax, %B52.endif.i.preheader ], [ %lsr.iv.next, %B52.endif.i ]
  %c.2.06.i = phi double [ %.239.i, %B52.endif.i ], [ 0.000000e+00, %B52.endif.i.preheader ]
  %.236.i = load double, double* %lsr.iv121, align 8, !noalias !14
  %.239.i = fadd double %c.2.06.i, %.236.i
  %lsr.iv.next = add nsw i64 %lsr.iv, -1
  %scevgep = getelementptr double, double* %lsr.iv121, i64 1
  %exitcond.not = icmp eq i64 %lsr.iv.next, 0
  br i1 %exitcond.not, label %_ZN5numba2np13old_arraymath9array_sum12_3clocals_3e14array_sum_implB3v12B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAE5ArrayIdLi1E1C7mutable7alignedE.exit.loopexit, label %B52.endif.i, !prof !0

_ZN5numba2np13old_arraymath9array_sum12_3clocals_3e14array_sum_implB3v12B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAE5ArrayIdLi1E1C7mutable7alignedE.exit.loopexit: ; preds = %B52.endif.i
  %phi.cast = fptosi double %.239.i to i64
  br label %B472.endif

common.ret:                                       ; preds = %B348.if, %B472.endif
  %common.ret.op = phi i32 [ 0, %B472.endif ], [ 1, %B348.if ]
  ret i32 %common.ret.op

B54.endif:                                        ; preds = %B54.endif.preheader, %B192.else.if
  %lsr.iv129 = phi i64 [ 10, %B54.endif.preheader ], [ %lsr.iv.next130, %B192.else.if ]
  %lsr.iv127 = phi i64 [ 0, %B54.endif.preheader ], [ %lsr.iv.next128, %B192.else.if ]
  %total_sum.4.0112 = phi double [ %total_sum.5.1, %B192.else.if ], [ 0.000000e+00, %B54.endif.preheader ]
  %max_value.3.0111 = phi double [ %max_value.4.1, %B192.else.if ], [ -1.000000e+07, %B54.endif.preheader ]
  %factorial_sum.3.0110 = phi i64 [ %factorial_sum.4.1, %B192.else.if ], [ 0, %B54.endif.preheader ]
  %count_total.3.0109 = phi i64 [ %count_total.4.1, %B192.else.if ], [ 0, %B54.endif.preheader ]
  %.44.0107 = phi i64 [ %.115, %B192.else.if ], [ 1, %B54.endif.preheader ]
  %1 = udiv i64 %.44.0107, 11
  %2 = mul nuw nsw i64 %1, 11
  %3 = add nsw i64 %2, -1
  %4 = udiv i64 %.44.0107, 7
  %5 = mul nuw nsw i64 %4, 7
  %6 = add nsw i64 %5, -1
  %7 = udiv i64 %.44.0107, 5
  %8 = mul nuw nsw i64 %7, 5
  %9 = add nsw i64 %8, -1
  %10 = add i64 %lsr.iv127, 1
  %.5.i = tail call { double, double, double, i64, i64 } @cfunc._ZN8__main__5func3B3v15B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx(i64 %lsr.iv129), !noalias !17
  %.5.elt.i = extractvalue { double, double, double, i64, i64 } %.5.i, 0
  %.5.elt2.i = extractvalue { double, double, double, i64, i64 } %.5.i, 1
  %.5.elt6.i = extractvalue { double, double, double, i64, i64 } %.5.i, 3
  %.5.elt8.i = extractvalue { double, double, double, i64, i64 } %.5.i, 4
  %.194 = fadd double %total_sum.4.0112, %.5.elt.i
  %.202 = fcmp ogt double %.5.elt2.i, %max_value.3.0111
  %max_value.4.1 = select i1 %.202, double %.5.elt2.i, double %max_value.3.0111
  %.220 = add nsw i64 %.5.elt6.i, %factorial_sum.3.0110
  %.230 = add nsw i64 %.5.elt8.i, %count_total.3.0109
  %.281 = icmp eq i64 %9, %lsr.iv127
  %.284 = fmul double %.194, 0x3FEE666666666666
  %total_sum.5.1 = select i1 %.281, double %.284, double %.194
  %.339 = icmp eq i64 %6, %lsr.iv127
  br i1 %.339, label %B182.else.if, label %B192.else.if

B182.else.if:                                     ; preds = %B54.endif
  %.362 = sdiv i64 %.220, 2
  %11 = and i64 %.220, -9223372036854775807
  %.365 = icmp eq i64 %11, -9223372036854775807
  %.372 = sext i1 %.365 to i64
  %spec.select92 = add nsw i64 %.362, %.372
  br label %B192.else.if

B192.else.if:                                     ; preds = %B182.else.if, %B54.endif
  %factorial_sum.4.1 = phi i64 [ %.220, %B54.endif ], [ %spec.select92, %B182.else.if ]
  %.434 = icmp eq i64 %3, %lsr.iv127
  %.191.i = mul nsw i64 %10, %10
  %.473 = select i1 %.434, i64 %.191.i, i64 0
  %count_total.4.1 = add nsw i64 %.230, %.473
  %lsr.iv.next128 = add nuw nsw i64 %lsr.iv127, 1
  %lsr.iv.next130 = add i64 %lsr.iv129, 10
  %.115 = add nuw i64 %.44.0107, 1
  %exitcond119.not = icmp eq i64 %0, %lsr.iv.next128
  br i1 %exitcond119.not, label %B228.else.if, label %B54.endif

B228.else.if:                                     ; preds = %B192.else.if, %B0.endif
  %count_total.3.0.lcssa = phi i64 [ 0, %B0.endif ], [ %count_total.4.1, %B192.else.if ]
  %factorial_sum.3.0.lcssa = phi i64 [ 0, %B0.endif ], [ %factorial_sum.4.1, %B192.else.if ]
  %max_value.3.0.lcssa = phi double [ -1.000000e+07, %B0.endif ], [ %max_value.4.1, %B192.else.if ]
  %total_sum.4.0.lcssa = phi double [ 0.000000e+00, %B0.endif ], [ %total_sum.5.1, %B192.else.if ]
  %.516 = sdiv i64 %arg.m, 2
  %12 = and i64 %arg.m, -9223372036854775807
  %.519 = icmp eq i64 %12, -9223372036854775807
  %.526 = sext i1 %.519 to i64
  %spec.select94 = add nsw i64 %.516, %.526
  %.618100.not = icmp slt i64 %spec.select94, 1
  br i1 %.618100.not, label %B348, label %B270.endif.preheader

B270.endif.preheader:                             ; preds = %B228.else.if
  br label %B270.endif

B270.endif:                                       ; preds = %B270.endif.preheader, %B270.endif
  %lsr.iv125 = phi i64 [ 5, %B270.endif.preheader ], [ %lsr.iv.next126, %B270.endif ]
  %lsr.iv123 = phi i64 [ %spec.select94, %B270.endif.preheader ], [ %lsr.iv.next124, %B270.endif ]
  %max_value.5.0104 = phi double [ %.698, %B270.endif ], [ %max_value.3.0.lcssa, %B270.endif.preheader ]
  %total_sum.6.0103 = phi double [ %.693, %B270.endif ], [ %total_sum.4.0.lcssa, %B270.endif.preheader ]
  %.5.i1 = tail call { double, double, double, i64, i64 } @cfunc._ZN8__main__5func2B3v10B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx(i64 %lsr.iv125), !noalias !20
  %.5.elt.i3 = extractvalue { double, double, double, i64, i64 } %.5.i1, 0
  %.5.elt2.i5 = extractvalue { double, double, double, i64, i64 } %.5.i1, 1
  %.693 = fadd double %total_sum.6.0103, %.5.elt.i3
  %.697 = fcmp ogt double %.5.elt2.i5, %max_value.5.0104
  %.698 = select i1 %.697, double %.5.elt2.i5, double %max_value.5.0104
  %lsr.iv.next124 = add i64 %lsr.iv123, -1
  %lsr.iv.next126 = add i64 %lsr.iv125, 5
  %exitcond118.not = icmp eq i64 %lsr.iv.next124, 0
  br i1 %exitcond118.not, label %B348, label %B270.endif

B348.if:                                          ; preds = %B0.endif.endif.endif.i.i, %B0.endif.endif.i.i, %B348
  %excinfo.1.0.ph.i = phi { i8*, i32, i8*, i8*, i32 }* [ @.const.picklebuf.4432610432, %B0.endif.endif.endif.i.i ], [ @.const.picklebuf.4433068480, %B0.endif.endif.i.i ], [ @.const.picklebuf.4433068544, %B348 ]
  store { i8*, i32, i8*, i8*, i32 }* %excinfo.1.0.ph.i, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  br label %common.ret

B348.endif.endif:                                 ; preds = %B0.endif.endif.endif.i.i
  %13 = icmp slt i64 %arg.m, 1
  %.5.i.i.i = getelementptr i8, i8* %.7.i.i.i.i, i64 24
  %14 = bitcast i8* %.5.i.i.i to double**
  %.6.i1.i.i = load double*, double** %14, align 8, !noalias !23
  %.26.i.i = shl nuw nsw i64 %arg.m, 3
  %.27.i.i = bitcast double* %.6.i1.i.i to i8*
  tail call void @llvm.memset.p0i8.i64(i8* align 1 %.27.i.i, i8 0, i64 %.26.i.i, i1 false), !noalias !24
  br i1 %13, label %B472.endif, label %B418.endif.preheader

B418.endif.preheader:                             ; preds = %B348.endif.endif
  br label %B418.endif

B418.endif:                                       ; preds = %B418.endif.preheader, %B418.endif
  %.774.098 = phi i64 [ %.843, %B418.endif ], [ 0, %B418.endif.preheader ]
  %.843 = add nuw nsw i64 %.774.098, 1
  %15 = add i64 %.774.098, 1
  %.5.i16 = tail call { double, double, double, i64, i64 } @cfunc._ZN8__main__5func1B2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx(i64 %15), !noalias !27
  %.5.elt.i18 = extractvalue { double, double, double, i64, i64 } %.5.i16, 0
  %scevgep122 = getelementptr double, double* %.6.i1.i.i, i64 %.774.098
  store double %.5.elt.i18, double* %scevgep122, align 8
  %exitcond117.not = icmp eq i64 %0, %.843
  br i1 %exitcond117.not, label %B52.endif.i.preheader, label %B418.endif

B472.endif:                                       ; preds = %B348.endif.endif, %_ZN5numba2np13old_arraymath9array_sum12_3clocals_3e14array_sum_implB3v12B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAE5ArrayIdLi1E1C7mutable7alignedE.exit.loopexit
  %c.2.0.lcssa.i = phi i64 [ %phi.cast, %_ZN5numba2np13old_arraymath9array_sum12_3clocals_3e14array_sum_implB3v12B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAE5ArrayIdLi1E1C7mutable7alignedE.exit.loopexit ], [ 0, %B348.endif.endif ]
  tail call void @NRT_decref(i8* nonnull %.7.i.i.i.i)
  %.993 = sitofp i64 %factorial_sum.3.0.lcssa to double
  %retptr.repack131 = bitcast { double, double, double, i64, i64 }* %retptr to double*
  store double %total_sum.6.0.lcssa, double* %retptr.repack131, align 8
  %retptr.repack54 = getelementptr inbounds { double, double, double, i64, i64 }, { double, double, double, i64, i64 }* %retptr, i64 0, i32 1
  store double %max_value.5.0.lcssa, double* %retptr.repack54, align 8
  %retptr.repack56 = getelementptr inbounds { double, double, double, i64, i64 }, { double, double, double, i64, i64 }* %retptr, i64 0, i32 2
  store double %.993, double* %retptr.repack56, align 8
  %retptr.repack58 = getelementptr inbounds { double, double, double, i64, i64 }, { double, double, double, i64, i64 }* %retptr, i64 0, i32 3
  store i64 %count_total.3.0.lcssa, i64* %retptr.repack58, align 8
  %retptr.repack60 = getelementptr inbounds { double, double, double, i64, i64 }, { double, double, double, i64, i64 }* %retptr, i64 0, i32 4
  store i64 %c.2.0.lcssa.i, i64* %retptr.repack60, align 8
  br label %common.ret
}

define i8* @_ZN7cpython8__main__5func4B3v17B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx(i8* nocapture readnone %py_closure, i8* %py_args, i8* nocapture readnone %py_kws) local_unnamed_addr {
entry:
  %.5 = alloca i8*, align 8
  %.6 = call i32 (i8*, i8*, i64, i64, ...) @PyArg_UnpackTuple(i8* %py_args, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.const.func4, i64 0, i64 0), i64 1, i64 1, i8** nonnull %.5)
  %.7 = icmp eq i32 %.6, 0
  %.36 = alloca { double, double, double, i64, i64 }, align 8
  %excinfo = alloca { i8*, i32, i8*, i8*, i32 }*, align 8
  store { i8*, i32, i8*, i8*, i32 }* null, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  br i1 %.7, label %common.ret, label %entry.endif, !prof !0

common.ret:                                       ; preds = %entry.endif.endif.endif.endif.endif.if.endif, %entry.endif.endif.endif.endif.endif.if.endif.if, %entry.endif.endif.endif.endif.endif.endif.endif.endif, %entry.endif.endif.endif, %entry, %entry.endif.endif.endif.endif.endif.if.if.if, %entry.endif.endif.endif.endif.if.endif, %entry.endif.if
  %common.ret.op = phi i8* [ null, %entry.endif.if ], [ %.60, %entry.endif.endif.endif.endif.if.endif ], [ null, %entry.endif.endif.endif.endif.endif.if.if.if ], [ null, %entry ], [ null, %entry.endif.endif.endif ], [ null, %entry.endif.endif.endif.endif.endif.endif.endif.endif ], [ null, %entry.endif.endif.endif.endif.endif.if.endif.if ], [ null, %entry.endif.endif.endif.endif.endif.if.endif ]
  ret i8* %common.ret.op

entry.endif:                                      ; preds = %entry
  %.11 = load i8*, i8** @_ZN08NumbaEnv8__main__5func4B3v17B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx, align 8
  %.16 = icmp eq i8* %.11, null
  br i1 %.16, label %entry.endif.if, label %entry.endif.endif, !prof !0

entry.endif.if:                                   ; preds = %entry.endif
  call void @PyErr_SetString(i8* nonnull @PyExc_RuntimeError, i8* getelementptr inbounds ([98 x i8], [98 x i8]* @".const.missing Environment: _ZN08NumbaEnv8__main__5func4B3v17B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx", i64 0, i64 0))
  br label %common.ret

entry.endif.endif:                                ; preds = %entry.endif
  %.20 = load i8*, i8** %.5, align 8
  %.23 = call i8* @PyNumber_Long(i8* %.20)
  %.24.not = icmp eq i8* %.23, null
  br i1 %.24.not, label %entry.endif.endif.endif, label %entry.endif.endif.if, !prof !0

entry.endif.endif.if:                             ; preds = %entry.endif.endif
  %.26 = call i64 @PyLong_AsLongLong(i8* nonnull %.23)
  call void @Py_DecRef(i8* nonnull %.23)
  br label %entry.endif.endif.endif

entry.endif.endif.endif:                          ; preds = %entry.endif.endif.if, %entry.endif.endif
  %.21.0 = phi i64 [ %.26, %entry.endif.endif.if ], [ 0, %entry.endif.endif ]
  %.31 = call i8* @PyErr_Occurred()
  %.32.not = icmp eq i8* %.31, null
  br i1 %.32.not, label %entry.endif.endif.endif.endif, label %common.ret, !prof !30

entry.endif.endif.endif.endif:                    ; preds = %entry.endif.endif.endif
  %0 = bitcast { double, double, double, i64, i64 }* %.36 to i8*
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(40) %0, i8 0, i64 40, i1 false)
  %.40 = call i32 @_ZN8__main__5func4B3v17B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx({ double, double, double, i64, i64 }* nonnull %.36, { i8*, i32, i8*, i8*, i32 }** nonnull %excinfo, i64 %.21.0) #2
  %.41 = load { i8*, i32, i8*, i8*, i32 }*, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  %.48 = icmp sgt i32 %.40, 0
  %.49 = select i1 %.48, { i8*, i32, i8*, i8*, i32 }* %.41, { i8*, i32, i8*, i8*, i32 }* undef
  switch i32 %.40, label %entry.endif.endif.endif.endif.endif [
    i32 -2, label %entry.endif.endif.endif.endif.if.endif
    i32 0, label %entry.endif.endif.endif.endif.if.endif
  ]

entry.endif.endif.endif.endif.endif:              ; preds = %entry.endif.endif.endif.endif
  %1 = icmp sgt i32 %.40, 0
  br i1 %1, label %entry.endif.endif.endif.endif.endif.if, label %entry.endif.endif.endif.endif.endif.endif.endif.endif

entry.endif.endif.endif.endif.if.endif:           ; preds = %entry.endif.endif.endif.endif, %entry.endif.endif.endif.endif
  %2 = bitcast { double, double, double, i64, i64 }* %.36 to double*
  %3 = bitcast { double, double, double, i64, i64 }* %.36 to i8*
  %sunkaddr = getelementptr inbounds i8, i8* %3, i64 32
  %4 = bitcast i8* %sunkaddr to i64*
  %.50.fca.4.load = load i64, i64* %4, align 8
  %5 = bitcast { double, double, double, i64, i64 }* %.36 to i8*
  %sunkaddr2 = getelementptr inbounds i8, i8* %5, i64 24
  %6 = bitcast i8* %sunkaddr2 to i64*
  %.50.fca.3.load = load i64, i64* %6, align 8
  %7 = bitcast { double, double, double, i64, i64 }* %.36 to i8*
  %sunkaddr3 = getelementptr inbounds i8, i8* %7, i64 16
  %8 = bitcast i8* %sunkaddr3 to double*
  %.50.fca.2.load = load double, double* %8, align 8
  %9 = bitcast { double, double, double, i64, i64 }* %.36 to i8*
  %sunkaddr4 = getelementptr inbounds i8, i8* %9, i64 8
  %10 = bitcast i8* %sunkaddr4 to double*
  %.50.fca.1.load = load double, double* %10, align 8
  %.50.fca.0.load = load double, double* %2, align 8
  %.60 = call i8* @PyTuple_New(i64 5)
  %.62 = call i8* @PyFloat_FromDouble(double %.50.fca.0.load)
  %.63 = call i32 @PyTuple_SetItem(i8* %.60, i64 0, i8* %.62)
  %.65 = call i8* @PyFloat_FromDouble(double %.50.fca.1.load)
  %.66 = call i32 @PyTuple_SetItem(i8* %.60, i64 1, i8* %.65)
  %.68 = call i8* @PyFloat_FromDouble(double %.50.fca.2.load)
  %.69 = call i32 @PyTuple_SetItem(i8* %.60, i64 2, i8* %.68)
  %.73 = call i8* @PyLong_FromLongLong(i64 %.50.fca.3.load)
  %.76 = call i32 @PyTuple_SetItem(i8* %.60, i64 3, i8* %.73)
  %.80 = call i8* @PyLong_FromLongLong(i64 %.50.fca.4.load)
  %.83 = call i32 @PyTuple_SetItem(i8* %.60, i64 4, i8* %.80)
  br label %common.ret

entry.endif.endif.endif.endif.endif.if:           ; preds = %entry.endif.endif.endif.endif.endif
  call void @PyErr_Clear()
  %.88 = load { i8*, i32, i8*, i8*, i32 }, { i8*, i32, i8*, i8*, i32 }* %.49, align 8
  %.89 = extractvalue { i8*, i32, i8*, i8*, i32 } %.88, 4
  %.90 = icmp sgt i32 %.89, 0
  %.93 = extractvalue { i8*, i32, i8*, i8*, i32 } %.88, 0
  %.95 = extractvalue { i8*, i32, i8*, i8*, i32 } %.88, 1
  br i1 %.90, label %entry.endif.endif.endif.endif.endif.if.if, label %entry.endif.endif.endif.endif.endif.if.else

entry.endif.endif.endif.endif.endif.if.if:        ; preds = %entry.endif.endif.endif.endif.endif.if
  %.96 = sext i32 %.95 to i64
  %.97 = call i8* @PyBytes_FromStringAndSize(i8* %.93, i64 %.96)
  %.98 = load { i8*, i32, i8*, i8*, i32 }, { i8*, i32, i8*, i8*, i32 }* %.49, align 8
  %.99 = extractvalue { i8*, i32, i8*, i8*, i32 } %.98, 2
  %.101 = extractvalue { i8*, i32, i8*, i8*, i32 } %.98, 3
  %.102 = bitcast i8* %.101 to i8* (i8*)*
  %.103 = call i8* %.102(i8* %.99)
  %.104 = icmp eq i8* %.103, null
  br i1 %.104, label %entry.endif.endif.endif.endif.endif.if.if.if, label %entry.endif.endif.endif.endif.endif.if.if.endif, !prof !0

entry.endif.endif.endif.endif.endif.if.else:      ; preds = %entry.endif.endif.endif.endif.endif.if
  %.117 = extractvalue { i8*, i32, i8*, i8*, i32 } %.88, 2
  %.118 = call i8* @numba_unpickle(i8* %.93, i32 %.95, i8* %.117)
  br label %entry.endif.endif.endif.endif.endif.if.endif

entry.endif.endif.endif.endif.endif.if.endif:     ; preds = %entry.endif.endif.endif.endif.endif.if.if.endif, %entry.endif.endif.endif.endif.endif.if.else
  %.120 = phi i8* [ %.108, %entry.endif.endif.endif.endif.endif.if.if.endif ], [ %.118, %entry.endif.endif.endif.endif.endif.if.else ]
  %.121.not = icmp eq i8* %.120, null
  br i1 %.121.not, label %common.ret, label %entry.endif.endif.endif.endif.endif.if.endif.if, !prof !0

entry.endif.endif.endif.endif.endif.if.if.if:     ; preds = %entry.endif.endif.endif.endif.endif.if.if
  call void @PyErr_SetString(i8* nonnull @PyExc_RuntimeError, i8* getelementptr inbounds ([61 x i8], [61 x i8]* @".const.Error creating Python tuple from runtime exception arguments", i64 0, i64 0))
  br label %common.ret

entry.endif.endif.endif.endif.endif.if.if.endif:  ; preds = %entry.endif.endif.endif.endif.endif.if.if
  %.108 = call i8* @numba_runtime_build_excinfo_struct(i8* %.97, i8* nonnull %.103)
  %.109 = bitcast { i8*, i32, i8*, i8*, i32 }* %.49 to i8*
  call void @NRT_Free(i8* nonnull %.109)
  br label %entry.endif.endif.endif.endif.endif.if.endif

entry.endif.endif.endif.endif.endif.if.endif.if:  ; preds = %entry.endif.endif.endif.endif.endif.if.endif
  call void @numba_do_raise(i8* nonnull %.120)
  br label %common.ret

entry.endif.endif.endif.endif.endif.endif.endif.endif: ; preds = %entry.endif.endif.endif.endif.endif
  call void @PyErr_SetString(i8* nonnull @PyExc_SystemError, i8* getelementptr inbounds ([43 x i8], [43 x i8]* @".const.unknown error when calling native function", i64 0, i64 0))
  br label %common.ret
}

declare i32 @PyArg_UnpackTuple(i8*, i8*, i64, i64, ...) local_unnamed_addr

declare void @PyErr_SetString(i8*, i8*) local_unnamed_addr

declare i8* @PyNumber_Long(i8*) local_unnamed_addr

declare i64 @PyLong_AsLongLong(i8*) local_unnamed_addr

declare void @Py_DecRef(i8*) local_unnamed_addr

declare i8* @PyErr_Occurred() local_unnamed_addr

declare i8* @PyTuple_New(i64) local_unnamed_addr

declare i8* @PyFloat_FromDouble(double) local_unnamed_addr

declare i32 @PyTuple_SetItem(i8*, i64, i8*) local_unnamed_addr

declare i8* @PyLong_FromLongLong(i64) local_unnamed_addr

declare void @PyErr_Clear() local_unnamed_addr

declare i8* @PyBytes_FromStringAndSize(i8*, i64) local_unnamed_addr

declare i8* @numba_unpickle(i8*, i32, i8*) local_unnamed_addr

declare i8* @numba_runtime_build_excinfo_struct(i8*, i8*) local_unnamed_addr

declare void @NRT_Free(i8*) local_unnamed_addr

declare void @numba_do_raise(i8*) local_unnamed_addr

define { double, double, double, i64, i64 } @cfunc._ZN8__main__5func4B3v17B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx(i64 %.1) local_unnamed_addr {
entry:
  %.3 = alloca { double, double, double, i64, i64 }, align 8
  %.fca.0.gep1 = bitcast { double, double, double, i64, i64 }* %.3 to double*
  %.fca.1.gep = getelementptr inbounds { double, double, double, i64, i64 }, { double, double, double, i64, i64 }* %.3, i64 0, i32 1
  %.fca.2.gep = getelementptr inbounds { double, double, double, i64, i64 }, { double, double, double, i64, i64 }* %.3, i64 0, i32 2
  %.fca.3.gep = getelementptr inbounds { double, double, double, i64, i64 }, { double, double, double, i64, i64 }* %.3, i64 0, i32 3
  %.fca.4.gep = getelementptr inbounds { double, double, double, i64, i64 }, { double, double, double, i64, i64 }* %.3, i64 0, i32 4
  %excinfo = alloca { i8*, i32, i8*, i8*, i32 }*, align 8
  %0 = bitcast { double, double, double, i64, i64 }* %.3 to i8*
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(40) %0, i8 0, i64 40, i1 false)
  store { i8*, i32, i8*, i8*, i32 }* null, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  %.7 = call i32 @_ZN8__main__5func4B3v17B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx({ double, double, double, i64, i64 }* nonnull %.3, { i8*, i32, i8*, i8*, i32 }** nonnull %excinfo, i64 %.1) #2
  %.8 = load { i8*, i32, i8*, i8*, i32 }*, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  %.9.not = icmp eq i32 %.7, 0
  %.15 = icmp sgt i32 %.7, 0
  %.16 = select i1 %.15, { i8*, i32, i8*, i8*, i32 }* %.8, { i8*, i32, i8*, i8*, i32 }* undef
  %.17.fca.0.load = load double, double* %.fca.0.gep1, align 8
  %.17.fca.1.load = load double, double* %.fca.1.gep, align 8
  %.17.fca.2.load = load double, double* %.fca.2.gep, align 8
  %.17.fca.3.load = load i64, i64* %.fca.3.gep, align 8
  %.17.fca.4.load = load i64, i64* %.fca.4.gep, align 8
  %inserted.f0 = insertvalue { double, double, double, i64, i64 } undef, double %.17.fca.0.load, 0
  %inserted.f1 = insertvalue { double, double, double, i64, i64 } %inserted.f0, double %.17.fca.1.load, 1
  %inserted.f2 = insertvalue { double, double, double, i64, i64 } %inserted.f1, double %.17.fca.2.load, 2
  %inserted.f3 = insertvalue { double, double, double, i64, i64 } %inserted.f2, i64 %.17.fca.3.load, 3
  %inserted.f4 = insertvalue { double, double, double, i64, i64 } %inserted.f3, i64 %.17.fca.4.load, 4
  %.24 = alloca i32, align 4
  store i32 0, i32* %.24, align 4
  br i1 %.9.not, label %common.ret, label %entry.if, !prof !30

entry.if:                                         ; preds = %entry
  %1 = icmp sgt i32 %.7, 0
  call void @numba_gil_ensure(i32* nonnull %.24)
  br i1 %1, label %entry.if.if, label %entry.if.endif.endif.endif

common.ret:                                       ; preds = %entry, %.27, %entry.if.if.if.if
  %common.ret.op = phi { double, double, double, i64, i64 } [ zeroinitializer, %entry.if.if.if.if ], [ %inserted.f4, %.27 ], [ %inserted.f4, %entry ]
  ret { double, double, double, i64, i64 } %common.ret.op

.27:                                              ; preds = %entry.if.if.endif, %entry.if.if.endif.if, %entry.if.endif.endif.endif
  %.75 = call i8* @PyUnicode_FromString(i8* getelementptr inbounds ([50 x i8], [50 x i8]* @".const.<numba.core.cpu.CPUContext object at 0x1089cca10>", i64 0, i64 0))
  call void @PyErr_WriteUnraisable(i8* %.75)
  call void @Py_DecRef(i8* %.75)
  call void @numba_gil_release(i32* nonnull %.24)
  br label %common.ret

entry.if.if:                                      ; preds = %entry.if
  call void @PyErr_Clear()
  %.30 = load { i8*, i32, i8*, i8*, i32 }, { i8*, i32, i8*, i8*, i32 }* %.16, align 8
  %.31 = extractvalue { i8*, i32, i8*, i8*, i32 } %.30, 4
  %.32 = icmp sgt i32 %.31, 0
  %.35 = extractvalue { i8*, i32, i8*, i8*, i32 } %.30, 0
  %.37 = extractvalue { i8*, i32, i8*, i8*, i32 } %.30, 1
  br i1 %.32, label %entry.if.if.if, label %entry.if.if.else

entry.if.if.if:                                   ; preds = %entry.if.if
  %.38 = sext i32 %.37 to i64
  %.39 = call i8* @PyBytes_FromStringAndSize(i8* %.35, i64 %.38)
  %.40 = load { i8*, i32, i8*, i8*, i32 }, { i8*, i32, i8*, i8*, i32 }* %.16, align 8
  %.41 = extractvalue { i8*, i32, i8*, i8*, i32 } %.40, 2
  %.43 = extractvalue { i8*, i32, i8*, i8*, i32 } %.40, 3
  %.44 = bitcast i8* %.43 to i8* (i8*)*
  %.45 = call i8* %.44(i8* %.41)
  %.46 = icmp eq i8* %.45, null
  br i1 %.46, label %entry.if.if.if.if, label %entry.if.if.if.endif, !prof !0

entry.if.if.else:                                 ; preds = %entry.if.if
  %.59 = extractvalue { i8*, i32, i8*, i8*, i32 } %.30, 2
  %.60 = call i8* @numba_unpickle(i8* %.35, i32 %.37, i8* %.59)
  br label %entry.if.if.endif

entry.if.if.endif:                                ; preds = %entry.if.if.if.endif, %entry.if.if.else
  %.62 = phi i8* [ %.50, %entry.if.if.if.endif ], [ %.60, %entry.if.if.else ]
  %.63.not = icmp eq i8* %.62, null
  br i1 %.63.not, label %.27, label %entry.if.if.endif.if, !prof !0

entry.if.if.if.if:                                ; preds = %entry.if.if.if
  call void @PyErr_SetString(i8* nonnull @PyExc_RuntimeError, i8* getelementptr inbounds ([61 x i8], [61 x i8]* @".const.Error creating Python tuple from runtime exception arguments.1", i64 0, i64 0))
  br label %common.ret

entry.if.if.if.endif:                             ; preds = %entry.if.if.if
  %.50 = call i8* @numba_runtime_build_excinfo_struct(i8* %.39, i8* nonnull %.45)
  %.51 = bitcast { i8*, i32, i8*, i8*, i32 }* %.16 to i8*
  call void @NRT_Free(i8* nonnull %.51)
  br label %entry.if.if.endif

entry.if.if.endif.if:                             ; preds = %entry.if.if.endif
  call void @numba_do_raise(i8* nonnull %.62)
  br label %.27

entry.if.endif.endif.endif:                       ; preds = %entry.if
  call void @PyErr_SetString(i8* nonnull @PyExc_SystemError, i8* getelementptr inbounds ([43 x i8], [43 x i8]* @".const.unknown error when calling native function.2", i64 0, i64 0))
  br label %.27
}

declare void @numba_gil_ensure(i32*) local_unnamed_addr

declare i8* @PyUnicode_FromString(i8*) local_unnamed_addr

declare void @PyErr_WriteUnraisable(i8*) local_unnamed_addr

declare void @numba_gil_release(i32*) local_unnamed_addr

declare { double, double, double, i64, i64 } @cfunc._ZN8__main__5func3B3v15B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx(i64) local_unnamed_addr

declare { double, double, double, i64, i64 } @cfunc._ZN8__main__5func2B3v10B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx(i64) local_unnamed_addr

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare { i64, i1 } @llvm.smul.with.overflow.i64(i64, i64) #0

declare noalias i8* @NRT_MemInfo_alloc_aligned(i64, i32) local_unnamed_addr

; Function Attrs: argmemonly mustprogress nocallback nofree nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #1

declare { double, double, double, i64, i64 } @cfunc._ZN8__main__5func1B2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx(i64) local_unnamed_addr

; Function Attrs: noinline
define linkonce_odr void @NRT_decref(i8* %.1) local_unnamed_addr #2 {
.3:
  %.4 = icmp eq i8* %.1, null
  br i1 %.4, label %common.ret1, label %.3.endif, !prof !0

common.ret1:                                      ; preds = %.3, %.3.endif
  ret void

.3.endif:                                         ; preds = %.3
  fence release
  %.8 = bitcast i8* %.1 to i64*
  %.4.i = atomicrmw sub i64* %.8, i64 1 monotonic, align 8
  %.10 = icmp eq i64 %.4.i, 1
  br i1 %.10, label %.3.endif.if, label %common.ret1, !prof !0

.3.endif.if:                                      ; preds = %.3.endif
  fence acquire
  tail call void @NRT_MemInfo_call_dtor(i8* nonnull %.1)
  ret void
}

declare void @NRT_MemInfo_call_dtor(i8*) local_unnamed_addr

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare i64 @llvm.smax.i64(i64, i64) #3

attributes #0 = { mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #1 = { argmemonly mustprogress nocallback nofree nounwind willreturn writeonly }
attributes #2 = { noinline }
attributes #3 = { nocallback nofree nosync nounwind readnone speculatable willreturn }

!0 = !{!"branch_weights", i32 1, i32 99}
!1 = !{!2, !4, !5, !7, !8, !10, !11, !13}
!2 = distinct !{!2, !3, !"_ZN5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %retptr"}
!3 = distinct !{!3, !"_ZN5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj"}
!4 = distinct !{!4, !3, !"_ZN5numba2np8arrayobj18_ol_array_allocate12_3clocals_3e4implB2v7B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %excinfo"}
!5 = distinct !{!5, !6, !"_ZN5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %retptr"}
!6 = distinct !{!6, !"_ZN5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj"}
!7 = distinct !{!7, !6, !"_ZN5numba2np8arrayobj15_call_allocatorB2v6B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAEN29typeref_5b_3cclass_20_27numba4core5types8npytypes14Array_27_3e_5dExj: %excinfo"}
!8 = distinct !{!8, !9, !"_ZN5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29: %retptr"}
!9 = distinct !{!9, !"_ZN5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29"}
!10 = distinct !{!10, !9, !"_ZN5numba2np8arrayobj11ol_np_empty12_3clocals_3e4implB2v5B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx39Function_28_3cclass_20_27float_27_3e_29: %excinfo"}
!11 = distinct !{!11, !12, !"_ZN5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29: %retptr"}
!12 = distinct !{!12, !"_ZN5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29"}
!13 = distinct !{!13, !12, !"_ZN5numba2np8arrayobj11ol_np_zeros12_3clocals_3e4implB2v4B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dEx48omitted_28default_3d_3cclass_20_27float_27_3e_29: %excinfo"}
!14 = !{!15}
!15 = distinct !{!15, !16, !"_ZN5numba2np13old_arraymath9array_sum12_3clocals_3e14array_sum_implB3v12B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAE5ArrayIdLi1E1C7mutable7alignedE: %retptr"}
!16 = distinct !{!16, !"_ZN5numba2np13old_arraymath9array_sum12_3clocals_3e14array_sum_implB3v12B42c8tJTC_2fWQA93W1AaAIYBPIqRBFCjDSZRVAJmaQIAE5ArrayIdLi1E1C7mutable7alignedE"}
!17 = !{!18}
!18 = distinct !{!18, !19, !"_ZN8__main__7__func3B3v16B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx: %retptr"}
!19 = distinct !{!19, !"_ZN8__main__7__func3B3v16B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx"}
!20 = !{!21}
!21 = distinct !{!21, !22, !"_ZN8__main__7__func2B3v14B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx: %retptr"}
!22 = distinct !{!22, !"_ZN8__main__7__func2B3v14B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx"}
!23 = !{!8, !10, !11, !13}
!24 = !{!25, !11, !13}
!25 = distinct !{!25, !26, !"_ZN5numba2np8arrayobj18ol_array_zero_fill12_3clocals_3e4implB2v8B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dE5ArrayIdLi1E1C7mutable7alignedE: %retptr"}
!26 = distinct !{!26, !"_ZN5numba2np8arrayobj18ol_array_zero_fill12_3clocals_3e4implB2v8B42c8tJTIeFIjxB2IKSgI4CrvQClcaMQ5hEEUSJJgA_3dE5ArrayIdLi1E1C7mutable7alignedE"}
!27 = !{!28}
!28 = distinct !{!28, !29, !"_ZN8__main__7__func1B2v9B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx: %retptr"}
!29 = distinct !{!29, !"_ZN8__main__7__func1B2v9B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEx"}
!30 = !{!"branch_weights", i32 99, i32 1}


"""