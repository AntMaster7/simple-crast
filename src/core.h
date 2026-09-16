/* core.h -- everything the simple renderer's files share: system includes,
   configuration, AVX2 math helpers, the core types, global state and the
   functions one file calls in another.

   The renderer is one translation unit (renderer.c includes every .c file),
   so "static" globals here are shared by all of them. Two other TUs: cpucheck.c
   holds the entry point and is compiled WITHOUT AVX2 code generation, so an
   older CPU gets a message box instead of an illegal-instruction crash, and
   image.c compiles the stb_image decoder.

   Portable C: SDL3 supplies the window, input, threads, atomics, aligned
   allocation and the inline / alignment keywords, so the same source builds
   with MSVC on Windows and GCC or Clang on Linux. */
#pragma once
#include <SDL3/SDL.h>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#ifdef _MSC_VER
#pragma warning(disable: 4244 4305)   /* double -> float in constants */
#endif

#define PATH_LEN 1024
#define SCENE_DIR "sponza/"        /* the folder holding Sponza.gltf, its .bin and textures */

/* The two compiler-specific spots. FORCE_INLINE is always static (SDL's
   SDL_FORCE_INLINE is static on GCC but not on MSVC); ctz32 is the index of
   the lowest set bit of a nonzero word. */
#if defined(_MSC_VER) && !defined(__clang__)
#define FORCE_INLINE static __forceinline
static __inline unsigned ctz32(uint32_t v) { unsigned long i; _BitScanForward(&i, v); return (unsigned)i; }
#else
#define FORCE_INLINE static inline __attribute__((always_inline))
static inline unsigned ctz32(uint32_t v) { return (unsigned)__builtin_ctz(v); }
#endif

/* ============================================================================
   Configuration
   ============================================================================ */
#define NUM_THREADS     8                /* thread pool size, main thread included */

#define TILE_SHIFT      6                /* 64x64-pixel tiles */
#define TILE_SIZE       (1 << TILE_SHIFT)

/* Span shape: one AVX2 register = 8 lanes = a 4x2 block of pixels. Lane L
   covers pixel (L % 4, L / 4) relative to the block's top-left corner. */
#define SPAN_W          4
#define SPAN_H          2
#define SPAN_BX         (TILE_SIZE / SPAN_W)   /* 16 blocks across a tile */
#define SPAN_BY         (TILE_SIZE / SPAN_H)   /* 32 blocks down a tile */
/* Tile-local depth is stored "swizzled": each 4x2 block is 8 contiguous floats. */
#define SPAN_OFF(bx, by) ((((size_t)(by) * SPAN_BX) + (size_t)(bx)) * 8)

/* Hierarchical raster classification: cells of 16x16 pixels = 4 blocks
   across x 8 block rows. Walks of at least HIER_MIN_BLOCKS blocks classify
   whole cells as empty / full from the edge functions' corner values. */
#define HIER_MIN_BLOCKS 48

#define MAX_MIPS        16
#define MAX_MATERIALS   64
#define VERT_CHUNK      4096             /* PH_VERTS job size, vertices */
#define TRI_CHUNK_SHIFT 10               /* PH_TRIS job size: 1024 triangles; also the bin rows */
#define TRI_CHUNK       (1 << TRI_CHUNK_SHIFT)
#define CLUSTER_SHIFT   6                /* 64 triangles per culling cluster */
#define CLUSTER_SIZE    (1 << CLUSTER_SHIFT)

#define CAM_FOV_DEG     70.0f
#define CAM_ZNEAR       0.05f
#define CAM_ZFAR        200.0f

/* Fixed-point edge functions (R3): screen positions snap to 1/256 px, the
   edge gradients are integer differences, and C holds the edge value at the
   centre of the bbox corner pixel with the top-left fill rule folded in. */
#define SUBPIX_SHIFT    8
#define SUBPIX_ONE      (1 << SUBPIX_SHIFT)
#define EDGE_SAT        (1 << 30)
#define EDGE_LANE_SPAN  15
#define EDGE_MAG_MAX    ((EDGE_SAT - 1) / EDGE_LANE_SPAN)
#define EDGE_COORD_MAX  (1 << 29)
#define EDGE_EXT_MAX    ((EDGE_MAG_MAX - 2) / SUBPIX_ONE)

/* frustum outcode bits */
#define OC_NEAR 1
#define OC_L    2
#define OC_R    4
#define OC_B    8
#define OC_T    16

/* ---- Lighting (linear, gamma-2 space), noon values of the full renderer ---- */
static const float LIT_sunDir[3]      = { 0.16f, 1.00f, 0.10f };  /* toward the sun, normalized at init */
static const float LIT_sunColor[3]    = { 2.60f, 2.30f, 1.90f };
static const float LIT_ambSky[3]      = { 0.30f, 0.38f, 0.55f };  /* hemisphere ambient, up */
static const float LIT_ambGround[3]   = { 0.22f, 0.19f, 0.16f };  /* hemisphere ambient, down */
static const float LIT_ambScale       = 0.55f;
static const float LIT_specKsMin      = 0.10f;
static const float LIT_shadowBias0    = 0.0025f;  /* receiver bias, normalized light depth */
static const float LIT_shadowBias1    = 0.0012f;  /* slope part, times tan(theta) capped at 6 */
static const float LIT_normalOffset   = 0.03f;    /* world units */
static const float LIT_shadowBiasScale   = 0.25f;
static const float LIT_normalOffsetScale = 0.25f;
static const float LIT_smSlopeTexels  = 2.5f;     /* bake: slope bias reach in texels */
static const float LIT_smSlopeCap     = 0.012f;   /* bake: cap on the slope bias */
static const float LIT_grazeLo        = 0.05f;    /* terminator fade: N.L below this, no sun */
static const float LIT_grazeHi        = 0.25f;

static const float SKY_zenith[3]      = { 0.11f, 0.26f, 0.60f };
static const float SKY_horizon[3]     = { 0.58f, 0.70f, 0.89f };
static const float SKY_ground[3]      = { 0.30f, 0.27f, 0.24f };
static const float SKY_sunColor[3]    = { 24.0f, 7.5f, 1.8f };
static const float SKY_sunRadiusDeg   = 1.5f;
static const float SKY_sunEdgeDeg     = 0.12f;
static const float SKY_haloPow        = 80.0f;
static const float SKY_haloGain       = 0.60f;
static const float SKY_glowPow        = 7.0f;
static const float SKY_glowGain       = 0.12f;
static const float SKY_cloudCoverLo   = 0.50f;
static const float SKY_cloudCoverHi   = 0.74f;
static const float SKY_cloudScale     = 1.0f / 55.0f;
static const float SKY_cloudHeight    = 28.0f;
#define CLOUD_RES 512                    /* power of two, tileable */

/* ---- Shadow map (R10): one static 8192^2 16-bit map, baked once at load ---- */
#define SM_SHIFT        13
#define SM_RES          (1 << SM_SHIFT)
#define SM_COARSE_SHIFT 4                /* 16x16-texel min/max cells */
#define SM_COARSE       (SM_RES >> SM_COARSE_SHIFT)
#define SM_QMAX         65535
#define SMT_SHIFT       7                /* 128x128-texel bake tiles */
#define SMT_SIZE        (1 << SMT_SHIFT)
#define SMT_N           (SM_RES >> SMT_SHIFT)   /* 64 tiles per axis */
#define SMT_TILES       (SMT_N * SMT_N)
#define SMT_CELLS       (SMT_SIZE >> SM_COARSE_SHIFT)

/* ============================================================================
   Small math
   ============================================================================ */
typedef struct { float x, y, z; } V3;
typedef struct { float x, y, z, w; } V4;
typedef struct { float m[4][4]; } M44;   /* row-major: clip = M * (x,y,z,1) */

FORCE_INLINE V3 v3(float x, float y, float z) { V3 r = { x, y, z }; return r; }
FORCE_INLINE V3 v3_add(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
FORCE_INLINE V3 v3_sub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
FORCE_INLINE V3 v3_scale(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
FORCE_INLINE float v3_dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
FORCE_INLINE V3 v3_cross(V3 a, V3 b) { return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
FORCE_INLINE V3 v3_norm(V3 a) { float l = sqrtf(v3_dot(a, a)); return l > 1e-20f ? v3_scale(a, 1.0f / l) : v3(0, 1, 0); }
FORCE_INLINE float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

/* ============================================================================
   AVX2 helpers. A lane is one pixel almost everywhere. Lane masks travel as
   all-ones / all-zeros vectors (AVX2 has no mask registers) and, where a
   branch or a bit trick needs them, as 8-bit movemasks.
   ============================================================================ */
#define VF __m256
#define VI __m256i

/* Mask bits -> mask vector: entry b holds -1 in every lane whose bit is set. */
static SDL_ALIGNED(32) int32_t g_maskLut[256][8];
FORCE_INLINE VI mask_vec(unsigned bits) { return _mm256_load_si256((const __m256i*)g_maskLut[bits & 255]); }
FORCE_INLINE int mask_bits(VF m) { return _mm256_movemask_ps(m); }
FORCE_INLINE int mask_bitsi(VI m) { return _mm256_movemask_ps(_mm256_castsi256_ps(m)); }

FORCE_INLINE VF v_rcp(VF x)          /* 1/x: rcpps seed + one Newton step */
{
    VF r = _mm256_rcp_ps(x);
    return _mm256_mul_ps(r, _mm256_fnmadd_ps(x, r, _mm256_set1_ps(2.0f)));
}
FORCE_INLINE VF v_rsqrt(VF x)        /* 1/sqrt(x): rsqrtps seed + one Newton step */
{
    VF r = _mm256_rsqrt_ps(x);
    VF h = _mm256_mul_ps(_mm256_set1_ps(0.5f), x);
    return _mm256_mul_ps(r, _mm256_fnmadd_ps(_mm256_mul_ps(h, r), r, _mm256_set1_ps(1.5f)));
}
/* floor(log2(x)) for positive, normal floats: the unbiased IEEE exponent.
   (AVX-512 has vgetexpps; AVX2 reads the bit field.) */
FORCE_INLINE VI v_exponent(VF x)
{
    VI bits = _mm256_castps_si256(x);
    return _mm256_sub_epi32(_mm256_and_si256(_mm256_srli_epi32(bits, 23), _mm256_set1_epi32(255)), _mm256_set1_epi32(127));
}
/* log2 via exponent + mantissa split: degree-5 fit of log2(1+f)/f on [0,1), max error 9e-6 */
FORCE_INLINE VF v_log2(VF x)
{
    VI bits = _mm256_castps_si256(x);
    VF e = _mm256_cvtepi32_ps(v_exponent(x));
    VF m = _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(bits, _mm256_set1_epi32(0x007FFFFF)),
                                               _mm256_set1_epi32(0x3F800000)));   /* mantissa in [1,2) */
    VF f = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));
    VF p = _mm256_set1_ps(-0.0345952f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.1464336f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(-0.3033897f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.4693017f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(-0.7204424f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.4426833f));
    return _mm256_fmadd_ps(p, f, e);
}
/* 2^x: degree-5 fit of 2^f on the fraction, times 2^floor(x) built as a float bit pattern */
FORCE_INLINE VF v_exp2(VF x)
{
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-126.0f)), _mm256_set1_ps(126.0f));
    VF xi = _mm256_floor_ps(x);
    VF f = _mm256_sub_ps(x, xi);
    VF p = _mm256_set1_ps(0.0018775767f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.0089893397f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.0558282287f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.2401596780f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.6931471806f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    VI e = _mm256_add_epi32(_mm256_cvtps_epi32(xi), _mm256_set1_epi32(127));
    VF scale = _mm256_castsi256_ps(_mm256_slli_epi32(e, 23));
    return _mm256_mul_ps(p, scale);
}
FORCE_INLINE VF v_pow(VF x, VF n)    /* x > 0 */
{
    return v_exp2(_mm256_mul_ps(n, v_log2(_mm256_max_ps(x, _mm256_set1_ps(1e-6f)))));
}
FORCE_INLINE float hmax8(VF v)
{
    __m128 q = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    q = _mm_max_ps(q, _mm_movehl_ps(q, q));
    q = _mm_max_ps(q, _mm_shuffle_ps(q, q, 1));
    return _mm_cvtss_f32(q);
}
FORCE_INLINE float hmin8(VF v)
{
    __m128 q = _mm_min_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    q = _mm_min_ps(q, _mm_movehl_ps(q, q));
    q = _mm_min_ps(q, _mm_shuffle_ps(q, q, 1));
    return _mm_cvtss_f32(q);
}
/* Gather horizontally adjacent element PAIRS: lane i reads the 8 bytes at
   base + idx[i]*scale as one 64-bit element. Returned split: lo = the low
   dword of each lane's qword, hi = the high dword. Two 4-lane qword gathers
   fetch 16 dwords -- half the gathered elements of four 8-lane dword gathers.
   Masked-off lanes read nothing and return 0. The element scale is a literal
   (the intrinsic needs a constant): 4 for dword texels, 2 for the 16-bit map. */
FORCE_INLINE void gather_pairs4(const void* base, VI idx, VI mask, VI* lo, VI* hi)
{
    const VI perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);   /* [l0 h0 l1 h1 ..] -> [l0 l1 l2 l3 | h0 h1 h2 h3] */
    const VI zero = _mm256_setzero_si256();
    VI ga = _mm256_mask_i32gather_epi64(zero, (const long long*)base, _mm256_castsi256_si128(idx),
                                        _mm256_cvtepi32_epi64(_mm256_castsi256_si128(mask)), 4);
    VI gb = _mm256_mask_i32gather_epi64(zero, (const long long*)base, _mm256_extracti128_si256(idx, 1),
                                        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(mask, 1)), 4);
    ga = _mm256_permutevar8x32_epi32(ga, perm);
    gb = _mm256_permutevar8x32_epi32(gb, perm);
    *lo = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm256_castsi256_si128(ga)), _mm256_castsi256_si128(gb), 1);
    *hi = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm256_extracti128_si256(ga, 1)), _mm256_extracti128_si256(gb, 1), 1);
}
FORCE_INLINE void gather_pairs2(const void* base, VI idx, VI mask, VI* lo, VI* hi)
{
    const VI perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);   /* [l0 h0 l1 h1 ..] -> [l0 l1 l2 l3 | h0 h1 h2 h3] */
    const VI zero = _mm256_setzero_si256();
    VI ga = _mm256_mask_i32gather_epi64(zero, (const long long*)base, _mm256_castsi256_si128(idx),
                                        _mm256_cvtepi32_epi64(_mm256_castsi256_si128(mask)), 2);
    VI gb = _mm256_mask_i32gather_epi64(zero, (const long long*)base, _mm256_extracti128_si256(idx, 1),
                                        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(mask, 1)), 2);
    ga = _mm256_permutevar8x32_epi32(ga, perm);
    gb = _mm256_permutevar8x32_epi32(gb, perm);
    *lo = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm256_castsi256_si128(ga)), _mm256_castsi256_si128(gb), 1);
    *hi = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm256_extracti128_si256(ga, 1)), _mm256_extracti128_si256(gb, 1), 1);
}

/* ============================================================================
   Core types
   ============================================================================ */
typedef struct Texture {
    uint32_t* texels;                 /* every mip level, contiguous, ARGB, +64 B tail pad */
    int w0, h0, nmips;
    int32_t moff[MAX_MIPS];           /* texel offset of each level (padded with the last level) */
    int32_t mw[MAX_MIPS], mh[MAX_MIPS];
} Texture;

typedef struct Material {
    Texture* tex;                     /* base colour, sqrt-encoded */
    Texture* nrm;                     /* tangent-space normal map, linear bytes; NULL = none */
    float ks, shininess;              /* Blinn-Phong from the metallic-roughness average */
    int alphaMask;                    /* glTF MASK */
    float alphaRef;                   /* alpha cutoff */
} Material;

typedef struct Vertex { float px, py, pz, u, v, nx, ny, nz; } Vertex;   /* 32 bytes */
typedef struct ScrVert { float sx, sy, z, invW; } ScrVert;              /* screen x, y, depth, 1/w */

/* A triangle set up for rasterization. Edge k: e = A*(px-minx) + B*(py-miny)
   + C at WHOLE pixels, inside is e >= 0. Attribute planes are evaluated at
   local pixel-centre coordinates (x - minx + 0.5): value = c + gx*X + gy*Y.
   u, v, normal and world position are premultiplied by 1/w. */
typedef struct Tri {
    int32_t A[3], B[3];
    int64_t C[3];
    float zc, zgx, zgy;               /* depth plane */
    float minz;                       /* nearest vertex depth: the depth floor */
    int minx, miny, maxx, maxy;       /* inclusive pixel bbox, clamped to the frame */
    int masked;                       /* alpha-tested material */
    float wc, wgx, wgy;               /* 1/w */
    float uc, ugx, ugy;
    float vc, vgx, vgy;
    float nc[3], ngx[3], ngy[3];      /* normal/w, mirrored for back faces */
    float pc[3], pgx[3], pgy[3];      /* world position/w */
    int mat, src;                     /* material, source triangle index */
    int16_t tq[3], ts;                /* face tangent (1.15 fixed point) and bitangent sign */
} Tri;

/* One tile's share of one triangle chunk: Tri slot indices. Cleared lazily:
   a bin whose stamp is not the frame's is empty. */
typedef struct Bin { int* items; int count, cap, stamp; } Bin;

typedef struct Cluster {
    V3 c; float r;                    /* world bounding sphere */
    V3 axis; float cosT, sinT;        /* face-normal cone */
    uint8_t ok;                       /* bit 0: cone valid, bit 1: every triangle single-sided */
} Cluster;

/* shadow caster in light space (texel x, texel y, depth 0..1) */
typedef struct SmTri {
    float A[3], B[3], C[3];           /* edges at texel centres, raw (the raster adds the conservative expansion) */
    float dgx, dgy, dc;               /* depth plane */
    float dmin, dmax;                 /* the triangle's own depth range */
    float sm;                         /* slope bias, uncapped */
    short x0, y0, x1, y1;             /* bbox in texels, widened by 3 */
} SmTri;
typedef struct SmBin { int* items; int count, cap; } SmBin;

typedef struct SlotCache { int next, end; char pad[56]; } SlotCache;

/* ============================================================================
   Global state
   ============================================================================ */
static char g_assetDir[PATH_LEN];     /* directory holding SCENE_DIR, with trailing separator */

/* scene */
static Vertex*   g_verts;   static int g_nverts;
static float*    g_vpx, *g_vpy, *g_vpz;       /* SoA positions, padded to 8 */
static uint32_t* g_idx;     static int g_ntris;
static int       g_ntriChunks;
static uint16_t* g_triMat;
static uint8_t*  g_triCullBack;      /* single-sided material */
static uint8_t*  g_triNoCast;        /* alpha-masked: no shadow */
static Material  g_mats[MAX_MATERIALS]; static int g_nmats;
static V3        g_aabbMin, g_aabbMax;
static uint8_t*  g_cloudDensity;     /* CLOUD_RES^2 + 64 B pad */

/* frame targets */
static int g_width, g_height, g_tilesX, g_tilesY, g_ntiles;
static float g_halfWf, g_halfHf;
static uint32_t* g_frame;             /* BGRA, top-down, row pitch g_frameStride */
static int g_frameStride;             /* g_tilesX * TILE_SIZE: every tile row starts on a 32-byte boundary */

/* per-frame transform and setup */
static V4*      g_clip;
static ScrVert* g_scr;
static uint8_t* g_oc;
static Tri*     g_tris;  static SDL_AtomicInt g_triCount; static int g_triCap;
static SlotCache g_slot[NUM_THREADS];
static Bin*     g_bins;               /* [chunk * g_ntiles + tile] */
static int      g_frameStamp = 1;
static SDL_AtomicInt* g_tileChunkBits;   /* [tile][g_chunkWords], 32 bits a word: chunks that binned into the tile */
static int      g_chunkWords;
static int*      g_tileOrder;         /* dispatch order, costliest tile first */
static unsigned* g_tileCost;          /* last frame's cycles per tile */

/* cluster culling */
static Cluster* g_clusters; static int g_nclusters;
static uint8_t* g_cluCull;

/* camera */
static V3 g_camPos; static float g_camYaw, g_camPitch;
static V3 g_fwd, g_rightv, g_upv;
static float g_moveSpeed;
static M44 g_vp;
static V3 g_rayA, g_rayB, g_rayC;     /* sky ray at pixel (x, y) = A + B*x + C*y */

/* lighting */
static V3 g_sunDir;
static float g_sunCosOuter, g_sunEdgeInv;   /* sun disc rim as cosines */
static float g_cloudPlaneY;

/* shadow map */
static uint16_t* g_smDepth;           /* SM_RES^2 quantized depths + 64 B pad */
static float* g_smCoarseMin, *g_smCoarseMax;
static float g_smRow[3][4];           /* world -> (texel x, texel y, depth 0..1) */
static float g_smBias0, g_smBias1;

/* thread pool phases */
enum { PH_VERTS, PH_TRIS, PH_TILES, PH_NRM, PH_SMVERTS, PH_SMSETUP, PH_SMTILES };

/* ============================================================================
   Functions other files call
   ============================================================================ */
/* pool.c */
static void pool_init(void);
static void dispatch(int phase, int total);
/* scene.c */
static void load_gltf(void);
static void job_nrm(int idx);
static void bake_clouds(void);
/* shadow.c */
static void shadow_bake(void);
static void job_sm_verts(int chunk);
static void job_sm_setup(int chunk);
static void job_sm_tile(int tile);
/* geometry.c */
static void job_verts(int chunk);
static void job_tris(int chunk, int tid);
static void clusters_build(void);
static void clusters_classify(void);
/* raster.c */
static void job_tile(int tile);
/* app.c */
static void camera_frame_constants(void);
