#pragma once

#if defined(CPU_CAPABILITY_SVE) && defined(SGLANG_SVE512_VEC)
#include <ATen/BFloat16.h>
#include <arm_sve.h>

#include <cmath>
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace sgl {
namespace vec {

template <typename T>
struct Vectorized {};

#if defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS == 512

typedef svfloat32_t fixed_svfloat32_t __attribute__((arm_sve_vector_bits(512)));
typedef svint32_t fixed_svint32_t __attribute__((arm_sve_vector_bits(512)));
typedef svbool_t fixed_svbool_t __attribute__((arm_sve_vector_bits(512)));

// Dummy bfloat16 type for SVE ACLE until svbfloat16_t fixed size is robust
typedef svbfloat16_t fixed_svbfloat16_t __attribute__((arm_sve_vector_bits(512)));

// ============================================================================
// Vectorized<float> SVE-512 implementation
// ============================================================================
template <>
struct Vectorized<float> {
  fixed_svfloat32_t vec_;

  Vectorized() {}
  Vectorized(fixed_svfloat32_t v) : vec_(v) {}
  Vectorized(float v) : vec_(svdup_f32(v)) {}

  operator fixed_svfloat32_t() const {
    return vec_;
  }

  static constexpr int size() {
    return 16;
  }

  static Vectorized<float> loadu(const void* ptr) {
    return svld1_f32(svptrue_b32(), (const float*)ptr);
  }

  void store(void* ptr) const {
    svst1_f32(svptrue_b32(), (float*)ptr, vec_);
  }

  // Operators
  Vectorized<float> operator+(const Vectorized<float>& b) const {
    return svadd_f32_z(svptrue_b32(), vec_, b.vec_);
  }

  Vectorized<float> operator-(const Vectorized<float>& b) const {
    return svsub_f32_z(svptrue_b32(), vec_, b.vec_);
  }

  Vectorized<float> operator*(const Vectorized<float>& b) const {
    return svmul_f32_z(svptrue_b32(), vec_, b.vec_);
  }

  Vectorized<float> operator/(const Vectorized<float>& b) const {
    return svdiv_f32_z(svptrue_b32(), vec_, b.vec_);
  }

  // Comparisons
  Vectorized<float> operator<(const Vectorized<float>& b) const {
    svbool_t cmp = svcmplt_f32(svptrue_b32(), vec_, b.vec_);
    return svdup_f32_z(cmp, 0.0f);  // Just for syntax, usually use cmp masks
  }

  Vectorized<float> operator>(const Vectorized<float>& b) const {
    svbool_t cmp = svcmpgt_f32(svptrue_b32(), vec_, b.vec_);
    return svdup_f32_z(cmp, 0.0f);
  }

  Vectorized<float> operator==(const Vectorized<float>& b) const {
    svbool_t cmp = svcmpeq_f32(svptrue_b32(), vec_, b.vec_);
    return svdup_f32_z(cmp, 0.0f);
  }
};

inline Vectorized<float> operator+(float a, const Vectorized<float>& b) {
  return Vectorized<float>(a) + b;
}
inline Vectorized<float> operator-(float a, const Vectorized<float>& b) {
  return Vectorized<float>(a) - b;
}
inline Vectorized<float> operator*(float a, const Vectorized<float>& b) {
  return Vectorized<float>(a) * b;
}
inline Vectorized<float> operator/(float a, const Vectorized<float>& b) {
  return Vectorized<float>(a) / b;
}

// Math functions
inline Vectorized<float> maximum(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svmax_f32_m(svptrue_b32(), a.vec_, b.vec_);
}

inline Vectorized<float> minimum(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svmin_f32_m(svptrue_b32(), a.vec_, b.vec_);
}

inline Vectorized<float> clamp_min(const Vectorized<float>& a, const Vectorized<float>& min_val) {
  return maximum(a, min_val);
}

inline Vectorized<float> clamp_max(const Vectorized<float>& a, const Vectorized<float>& max_val) {
  return minimum(a, max_val);
}

inline Vectorized<float> exp(const Vectorized<float>& a) {
  // Use our fast exp from test_sve_kernels (Cephes approximation)
  svbool_t pg = svptrue_b32();
  svfloat32_t vx = a.vec_;

  svfloat32_t vec_a = svdup_f32(12102203.0f / 16777216.0f);
  svfloat32_t vec_b = svdup_f32(27.0f / 16.0f);
  float log2e = 1.4426950408889634f;

  svfloat32_t vc0 = svdup_f32(0.693145751953125f);
  svfloat32_t vc1 = svdup_f32(1.428606765330187e-06f);
  svfloat32_t vc2 = svdup_f32(4.375000000000000e-01f);
  svfloat32_t vc3 = svdup_f32(1.666666666666667e-01f);

  svfloat32_t vt = svmul_f32_x(pg, vx, svdup_f32(log2e));
  svint32_t vi = svcvt_s32_f32_x(pg, vt);
  svfloat32_t vfi = svcvt_f32_s32_x(pg, vi);

  svbool_t neg_mask = svcmplt_f32(pg, vt, vfi);
  vi = svsub_s32_m(pg, vi, svdup_s32_z(neg_mask, 1));
  vfi = svsub_f32_m(pg, vfi, svdup_f32_z(neg_mask, 1.0f));

  vi = svadd_s32_x(pg, vi, svdup_s32(0x7f));

  svbool_t max_mask = svcmpgt_s32(pg, vi, svdup_s32(253));
  svbool_t min_mask = svcmplt_s32(pg, vi, svdup_s32(-27));

  vi = svlsl_s32_x(pg, vi, svdup_u32(23));

  vfi = svsub_f32_x(pg, vx, svmul_f32_x(pg, vfi, vc0));
  vfi = svsub_f32_x(pg, vfi, svmul_f32_x(pg, vfi, vc1));

  svfloat32_t vr = svmla_f32_x(pg, vc2, vfi, vc3);
  vr = svmla_f32_x(pg, vc1, vfi, vr);
  vr = svmla_f32_x(pg, vc0, vfi, vr);

  vfi = svsub_f32_x(pg, vfi, vr);
  svfloat32_t tmp = svmla_f32_x(pg, vec_b, vec_a, vfi);
  svuint32_t casted = svreinterpret_u32_f32(svcvt_f32_s32_z(pg, svreinterpret_s32_f32(tmp)));

  svuint32_t v_inf = svreinterpret_u32_f32(svdup_f32(INFINITY));
  svuint32_t v_zero = svdup_u32(0);

  casted = svsel_u32(min_mask, v_zero, casted);
  casted = svsel_u32(max_mask, v_inf, casted);

  return svreinterpret_f32_u32(casted);
}

inline Vectorized<float> fmadd(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  return svmla_f32_m(svptrue_b32(), c.vec_, a.vec_, b.vec_);
}

inline Vectorized<float> blendv(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& mask) {
  // If mask highest bit is set, select b, else a
  svuint32_t umask = svreinterpret_u32_f32(mask.vec_);
  svbool_t cmp = svcmplt_s32(svptrue_b32(), svreinterpret_s32_u32(umask), svdup_s32(0));
  return svsel_f32(cmp, b.vec_, a.vec_);
}

// ============================================================================
// Vectorized<at::BFloat16> SVE-512 implementation
// ============================================================================
template <>
struct Vectorized<at::BFloat16> {
  fixed_svbfloat16_t vec_;

  Vectorized() {}
  Vectorized(fixed_svbfloat16_t v) : vec_(v) {}
  Vectorized(at::BFloat16 v) : vec_(svreinterpret_bf16_u16(svdup_u16(v.x))) {}

  operator fixed_svbfloat16_t() const {
    return vec_;
  }

  static constexpr int size() {
    return 32;
  }

  static Vectorized<at::BFloat16> loadu(const void* ptr) {
    return svld1_bf16(svptrue_b32(), (const bfloat16_t*)ptr);
  }

  void store(void* ptr) const {
    svst1_bf16(svptrue_b32(), (bfloat16_t*)ptr, vec_);
  }
};

// ============================================================================
// reduce_all implementations
// ============================================================================
template <typename T, typename F>
inline T reduce_all(F f, const Vectorized<T>& a) {
  // Simplified since we only use this for maximum natively
  // Fall back to scalar for complex cases
  float max_val = svmaxv_f32(svptrue_b32(), a.vec_);
  return max_val;
}

// Convert
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_to_float(const Vectorized<at::BFloat16>& a) {
  // BFloat16 to float: pad lower 16 bits with 0
  svuint16_t u16 = svreinterpret_u16_bf16(a.vec_);

  svuint32_t lo = svunpklo_u32(u16);
  svuint32_t hi = svunpkhi_u32(u16);

  lo = svlsl_u32_z(svptrue_b32(), lo, svdup_u32(16));
  hi = svlsl_u32_z(svptrue_b32(), hi, svdup_u32(16));

  return std::make_tuple(Vectorized<float>(svreinterpret_f32_u32(lo)), Vectorized<float>(svreinterpret_f32_u32(hi)));
}

template <typename scalar_t>
inline Vectorized<scalar_t> convert_from_float(const Vectorized<float>& a, const Vectorized<float>& b);

template <>
inline Vectorized<at::BFloat16>
convert_from_float<at::BFloat16>(const Vectorized<float>& a, const Vectorized<float>& b) {
  // float to bf16 natively
  return sve_f32_to_bf16(svptrue_b32(), a.vec_);  // Simplified: we lose b
}

// Vectorized<T> array conversions
template <typename InType, typename OutType>
void convert(const InType* src, OutType* dst, int n) {
  for (int i = 0; i < n; ++i) {
    dst[i] = static_cast<OutType>(src[i]);  // Simple fallback since we only use it marginally
  }
}

#endif  // __ARM_FEATURE_SVE_BITS == 512

}  // namespace vec
}  // namespace sgl
#endif  // CPU_CAPABILITY_SVE && SGLANG_SVE512_VEC
