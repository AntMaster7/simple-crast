/* shadow.c -- sun shadows (R10): one 8192^2 orthographic 16-bit depth map for
   the fixed noon sun, baked once at load on the thread pool, and the 3x3 tent
   PCF sampler the shade pass calls.

   The bake rules each fixed a visible artifact in the full renderer:
   - casting is double-sided (only alpha-masked materials do not cast);
   - conservative raster: edges pushed out 0.75 * (|A| + |B|) texels;
   - interpolated depth clamped to the triangle's own vertex depth range;
   - a slope bias baked into the map, capped;
   - depth is a MIN, so neither tile order nor thread order changes the map.

   Phases: PH_SMVERTS (8-wide light-space transform), PH_SMSETUP (per
   1024-triangle chunk: caster records binned by (bake tile, chunk), so each
   bin has one writer), PH_SMTILES (one 128x128-texel tile rasterized into a
   stack buffer, quantized into the map, its 16x16 min/max cells reduced). */
#include "core.h"

static float* g_lsX, *g_lsY, *g_lsD;   /* light-space vertices, padded to 8 */
static SmTri* g_smTris;
static SmBin* g_smBins;                /* [tile * g_ntriChunks + chunk] */

/* ortho along -sunDir fitted to the scene AABB with a 1% margin */
static void shadow_basis(V3 sunDir, float row[3][4])
{
    V3 F = v3_scale(sunDir, -1.0f);
    V3 seed = fabsf(F.x) < 0.9f ? v3(1, 0, 0) : v3(0, 0, 1);
    V3 R = v3_norm(v3_cross(seed, F));
    V3 U = v3_cross(F, R);
    float rMin = 1e30f, rMax = -1e30f, uMin = 1e30f, uMax = -1e30f, dMin = 1e30f, dMax = -1e30f;
    for (int i = 0; i < 8; i++) {
        V3 c = v3(i & 1 ? g_aabbMax.x : g_aabbMin.x, i & 2 ? g_aabbMax.y : g_aabbMin.y, i & 4 ? g_aabbMax.z : g_aabbMin.z);
        float r = v3_dot(c, R), u = v3_dot(c, U), d = v3_dot(c, F);
        if (r < rMin) rMin = r; if (r > rMax) rMax = r;
        if (u < uMin) uMin = u; if (u > uMax) uMax = u;
        if (d < dMin) dMin = d; if (d > dMax) dMax = d;
    }
    float mr = (rMax - rMin) * 0.01f, mu = (uMax - uMin) * 0.01f, md = (dMax - dMin) * 0.01f;
    rMin -= mr; rMax += mr; uMin -= mu; uMax += mu; dMin -= md; dMax += md;
    float sr = (float)SM_RES / (rMax - rMin), su = (float)SM_RES / (uMax - uMin), sd = 1.0f / (dMax - dMin);
    row[0][0] = R.x * sr; row[0][1] = R.y * sr; row[0][2] = R.z * sr; row[0][3] = -rMin * sr;
    row[1][0] = U.x * su; row[1][1] = U.y * su; row[1][2] = U.z * su; row[1][3] = -uMin * su;
    row[2][0] = F.x * sd; row[2][1] = F.y * sd; row[2][2] = F.z * sd; row[2][3] = -dMin * sd;
}

static void job_sm_verts(int chunk)
{
    int i0 = chunk * VERT_CHUNK, i1 = i0 + VERT_CHUNK;
    if (i1 > g_nverts) i1 = g_nverts;
    const float (*row)[4] = g_smRow;
    VF r00 = _mm256_set1_ps(row[0][0]), r01 = _mm256_set1_ps(row[0][1]), r02 = _mm256_set1_ps(row[0][2]), r03 = _mm256_set1_ps(row[0][3]);
    VF r10 = _mm256_set1_ps(row[1][0]), r11 = _mm256_set1_ps(row[1][1]), r12 = _mm256_set1_ps(row[1][2]), r13 = _mm256_set1_ps(row[1][3]);
    VF r20 = _mm256_set1_ps(row[2][0]), r21 = _mm256_set1_ps(row[2][1]), r22 = _mm256_set1_ps(row[2][2]), r23 = _mm256_set1_ps(row[2][3]);
    for (int i = i0; i < i1; i += 8) {
        VF x = _mm256_load_ps(g_vpx + i), y = _mm256_load_ps(g_vpy + i), z = _mm256_load_ps(g_vpz + i);
        _mm256_store_ps(g_lsX + i, _mm256_fmadd_ps(x, r00, _mm256_fmadd_ps(y, r01, _mm256_fmadd_ps(z, r02, r03))));
        _mm256_store_ps(g_lsY + i, _mm256_fmadd_ps(x, r10, _mm256_fmadd_ps(y, r11, _mm256_fmadd_ps(z, r12, r13))));
        _mm256_store_ps(g_lsD + i, _mm256_fmadd_ps(x, r20, _mm256_fmadd_ps(y, r21, _mm256_fmadd_ps(z, r22, r23))));
    }
}

static void smbin_push(SmBin* b, int v)
{
    if (b->count == b->cap) {
        b->cap = b->cap ? b->cap * 2 : 32;
        b->items = (int*)realloc(b->items, (size_t)b->cap * sizeof(int));
    }
    b->items[b->count++] = v;
}

static void job_sm_setup(int chunk)
{
    int t0 = chunk << TRI_CHUNK_SHIFT, t1 = t0 + TRI_CHUNK;
    if (t1 > g_ntris) t1 = g_ntris;
    for (int t = t0; t < t1; t++) {
        if (g_triNoCast[t]) continue;             /* cut-outs would cast their whole quads */
        const int i0 = g_idx[t * 3], i1 = g_idx[t * 3 + 1], i2 = g_idx[t * 3 + 2];
        float x0 = g_lsX[i0], y0 = g_lsY[i0], d0 = g_lsD[i0];
        float x1 = g_lsX[i1], y1 = g_lsY[i1], d1 = g_lsD[i1];
        float x2 = g_lsX[i2], y2 = g_lsY[i2], d2 = g_lsD[i2];
        float area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
        if (area2 > -1e-8f && area2 < 1e-8f) continue;
        if (area2 > 0) {                          /* double-sided: reorient to the inside >= 0 convention */
            float tf = x1; x1 = x2; x2 = tf;
            tf = y1; y1 = y2; y2 = tf;
            tf = d1; d1 = d2; d2 = tf;
            area2 = -area2;
        }
        SmTri* r = &g_smTris[t];
        const float invA = 1.0f / area2;
        r->dgx = ((d1 - d0) * (y2 - y0) - (d2 - d0) * (y1 - y0)) * invA;
        r->dgy = ((d2 - d0) * (x1 - x0) - (d1 - d0) * (x2 - x0)) * invA;
        r->dc = d0 - r->dgx * x0 - r->dgy * y0;
        r->dmin = d0 < d1 ? (d0 < d2 ? d0 : d2) : (d1 < d2 ? d1 : d2);
        r->dmax = d0 > d1 ? (d0 > d2 ? d0 : d2) : (d1 > d2 ? d1 : d2);
        r->sm = LIT_smSlopeTexels * (fabsf(r->dgx) + fabsf(r->dgy));
        r->A[0] = y2 - y1; r->B[0] = x1 - x2; r->C[0] = -r->A[0] * x1 - r->B[0] * y1;
        r->A[1] = y0 - y2; r->B[1] = x2 - x0; r->C[1] = -r->A[1] * x2 - r->B[1] * y2;
        r->A[2] = y1 - y0; r->B[2] = x0 - x1; r->C[2] = -r->A[2] * x0 - r->B[2] * y0;
        float mnx = x0 < x1 ? (x0 < x2 ? x0 : x2) : (x1 < x2 ? x1 : x2);
        float mxx = x0 > x1 ? (x0 > x2 ? x0 : x2) : (x1 > x2 ? x1 : x2);
        float mny = y0 < y1 ? (y0 < y2 ? y0 : y2) : (y1 < y2 ? y1 : y2);
        float mxy = y0 > y1 ? (y0 > y2 ? y0 : y2) : (y1 > y2 ? y1 : y2);
        int ix0 = (int)mnx - 3, ix1 = (int)mxx + 3, iy0 = (int)mny - 3, iy1 = (int)mxy + 3;
        if (ix0 < 0) ix0 = 0;
        if (iy0 < 0) iy0 = 0;
        if (ix1 > SM_RES - 1) ix1 = SM_RES - 1;
        if (iy1 > SM_RES - 1) iy1 = SM_RES - 1;
        if (ix0 > ix1 || iy0 > iy1) continue;
        r->x0 = (short)ix0; r->y0 = (short)iy0; r->x1 = (short)ix1; r->y1 = (short)iy1;
        for (int ty = iy0 >> SMT_SHIFT; ty <= (iy1 >> SMT_SHIFT); ty++)
            for (int tx = ix0 >> SMT_SHIFT; tx <= (ix1 >> SMT_SHIFT); tx++)
                smbin_push(&g_smBins[(size_t)(ty * SMT_N + tx) * g_ntriChunks + chunk], t);
    }
}

/* Rasterize one bake tile's casters into buf (128x128 texels, origin tx0,ty0).
   Columns in 8-texel groups; for bboxes taller than a 16-row run, each run is
   first tested at its four corner texels and skipped when one edge excludes
   all four (an edge function is linear over the run). */
static void sm_raster_tile(float* buf, int tile, int tx0, int ty0)
{
    const VF lane = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
    const VF zero = _mm256_setzero_ps();
    for (int c = 0; c < g_ntriChunks; c++) {
        const SmBin* b = &g_smBins[(size_t)tile * g_ntriChunks + c];
        for (int k = 0; k < b->count; k++) {
            const SmTri* r = &g_smTris[b->items[k]];
            int lx0 = r->x0 - tx0, lx1 = r->x1 - tx0, ly0 = r->y0 - ty0, ly1 = r->y1 - ty0;
            if (lx0 < 0) lx0 = 0;
            if (ly0 < 0) ly0 = 0;
            if (lx1 > SMT_SIZE - 1) lx1 = SMT_SIZE - 1;
            if (ly1 > SMT_SIZE - 1) ly1 = SMT_SIZE - 1;
            if (lx0 > lx1 || ly0 > ly1) continue;
            float smd = r->sm; if (smd > LIT_smSlopeCap) smd = LIT_smSlopeCap;
            float Cx[3], mg[3];
            for (int e = 0; e < 3; e++) {
                const float reach = fabsf(r->A[e]) + fabsf(r->B[e]);
                Cx[e] = r->C[e] + 0.75f * reach;          /* the conservative expansion */
                mg[e] = -reach * (1.0f / 64.0f);          /* the run skip's margin, far above float rounding */
            }
            VF A0 = _mm256_set1_ps(r->A[0]), A1 = _mm256_set1_ps(r->A[1]), A2 = _mm256_set1_ps(r->A[2]);
            VF B0 = _mm256_set1_ps(r->B[0]), B1 = _mm256_set1_ps(r->B[1]), B2 = _mm256_set1_ps(r->B[2]);
            VF C0 = _mm256_set1_ps(Cx[0]), C1 = _mm256_set1_ps(Cx[1]), C2 = _mm256_set1_ps(Cx[2]);
            VF dgx = _mm256_set1_ps(r->dgx), dgy = _mm256_set1_ps(r->dgy), dc = _mm256_set1_ps(r->dc + smd);
            VF dmin = _mm256_set1_ps(r->dmin + smd), dmax = _mm256_set1_ps(r->dmax + smd);
            const int runs = (ly1 - ly0) >= 31;
            for (int xg = lx0 & ~7; xg <= lx1; xg += 8) {
                int lo = lx0 - xg, hi = lx1 - xg;
                if (lo < 0) lo = 0;
                if (hi > 7) hi = 7;
                const VF xm = _mm256_castsi256_ps(mask_vec(((1u << (hi - lo + 1)) - 1u) << lo));
                VF xc = _mm256_add_ps(_mm256_set1_ps((float)(tx0 + xg) + 0.5f), lane);   /* texel centres */
                VF e0x = _mm256_fmadd_ps(A0, xc, C0), e1x = _mm256_fmadd_ps(A1, xc, C1), e2x = _mm256_fmadd_ps(A2, xc, C2);
                VF dx = _mm256_fmadd_ps(dgx, xc, dc);
                const float xcl = (float)(tx0 + xg + lo) + 0.5f, xch = (float)(tx0 + xg + hi) + 0.5f;
                float* col = buf + xg;
                for (int yb = ly0; yb <= ly1; ) {
                    int ye = yb | 15;
                    if (ye > ly1) ye = ly1;
                    if (runs) {
                        const float ycb = (float)(ty0 + yb) + 0.5f, yce = (float)(ty0 + ye) + 0.5f;
                        const __m128 cxv = _mm_setr_ps(xcl, xch, xcl, xch), cyv = _mm_setr_ps(ycb, ycb, yce, yce);
                        int skip = 0;
                        for (int e = 0; e < 3 && !skip; e++) {
                            __m128 ev = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(r->A[e]), cxv), _mm_mul_ps(_mm_set1_ps(r->B[e]), cyv)), _mm_set1_ps(Cx[e]));
                            skip = _mm_movemask_ps(_mm_cmplt_ps(ev, _mm_set1_ps(mg[e]))) == 15;
                        }
                        if (skip) { yb = ye + 1; continue; }
                    }
                    for (int y = yb; y <= ye; y++) {
                        VF yc = _mm256_set1_ps((float)(ty0 + y) + 0.5f);
                        VF in = _mm256_and_ps(xm, _mm256_cmp_ps(_mm256_fmadd_ps(B0, yc, e0x), zero, _CMP_GE_OQ));
                        in = _mm256_and_ps(in, _mm256_cmp_ps(_mm256_fmadd_ps(B1, yc, e1x), zero, _CMP_GE_OQ));
                        in = _mm256_and_ps(in, _mm256_cmp_ps(_mm256_fmadd_ps(B2, yc, e2x), zero, _CMP_GE_OQ));
                        if (!_mm256_movemask_ps(in)) continue;
                        VF d = _mm256_min_ps(_mm256_max_ps(_mm256_fmadd_ps(dgy, yc, dx), dmin), dmax);
                        float* p = col + y * SMT_SIZE;
                        VF cur = _mm256_load_ps(p);
                        _mm256_store_ps(p, _mm256_blendv_ps(cur, _mm256_min_ps(cur, d), in));   /* keep the nearest caster */
                    }
                    yb = ye + 1;
                }
            }
        }
    }
}

/* 8 depths -> 16-bit map values, still as int32 lanes */
FORCE_INLINE VI sm_quant(VF v)
{
    VF q = _mm256_mul_ps(v, _mm256_set1_ps((float)SM_QMAX));
    q = _mm256_min_ps(_mm256_max_ps(q, _mm256_setzero_ps()), _mm256_set1_ps((float)SM_QMAX));
    return _mm256_cvtps_epi32(q);
}

static void job_sm_tile(int tile)
{
    const int tx0 = (tile % SMT_N) << SMT_SHIFT, ty0 = (tile / SMT_N) << SMT_SHIFT;
    SDL_ALIGNED(32) float buf[SMT_SIZE * SMT_SIZE];
    const VF farv = _mm256_set1_ps(1e9f);                /* "nothing casts here" quantizes to SM_QMAX */
    for (int i = 0; i < SMT_SIZE * SMT_SIZE; i += 8) _mm256_store_ps(buf + i, farv);
    sm_raster_tile(buf, tile, tx0, ty0);
    for (int y = 0; y < SMT_SIZE; y++) {                 /* quantize and stream out */
        uint16_t* dst = g_smDepth + (size_t)(ty0 + y) * SM_RES + tx0;
        for (int x = 0; x < SMT_SIZE; x += 8) {
            VI q = sm_quant(_mm256_load_ps(buf + y * SMT_SIZE + x));
            VI p = _mm256_permute4x64_epi64(_mm256_packus_epi32(q, q), 0xD8);   /* 8 words in lane order in the low half */
            _mm_stream_si128((__m128i*)(dst + x), _mm256_castsi256_si128(p));
        }
    }
    _mm_sfence();
    for (int cy = 0; cy < SMT_CELLS; cy++)               /* 16x16-texel min/max cells, over the STORED values */
        for (int cx = 0; cx < SMT_CELLS; cx++) {
            VF mn = _mm256_set1_ps(1e30f), mx = _mm256_set1_ps(-1e30f);
            for (int rr = 0; rr < 16; rr++)
                for (int x = 0; x < 16; x += 8) {
                    VF q = _mm256_cvtepi32_ps(sm_quant(_mm256_load_ps(buf + ((cy << 4) + rr) * SMT_SIZE + (cx << 4) + x)));
                    mn = _mm256_min_ps(mn, q);
                    mx = _mm256_max_ps(mx, q);
                }
            const size_t ci = (size_t)((ty0 >> 4) + cy) * SM_COARSE + (size_t)((tx0 >> 4) + cx);
            g_smCoarseMin[ci] = hmin8(mn);
            g_smCoarseMax[ci] = hmax8(mx);
        }
}

static void shadow_bake(void)
{
    shadow_basis(g_sunDir, g_smRow);
    g_smBias0 = LIT_shadowBias0 * LIT_shadowBiasScale;
    g_smBias1 = LIT_shadowBias1 * LIT_shadowBiasScale;

    g_smDepth = (uint16_t*)SDL_aligned_alloc(64, (size_t)SM_RES * SM_RES * 2 + 64);
    memset((uint8_t*)g_smDepth + (size_t)SM_RES * SM_RES * 2, 0xFF, 64);
    g_smCoarseMin = (float*)SDL_aligned_alloc(64, (size_t)SM_COARSE * SM_COARSE * 4);
    g_smCoarseMax = (float*)SDL_aligned_alloc(64, (size_t)SM_COARSE * SM_COARSE * 4);
    size_t nv8 = ((size_t)g_nverts + 7) & ~(size_t)7;
    g_lsX = (float*)SDL_aligned_alloc(32, nv8 * 4);
    g_lsY = (float*)SDL_aligned_alloc(32, nv8 * 4);
    g_lsD = (float*)SDL_aligned_alloc(32, nv8 * 4);
    g_smTris = (SmTri*)malloc((size_t)g_ntris * sizeof(SmTri));
    g_smBins = (SmBin*)calloc((size_t)SMT_TILES * g_ntriChunks, sizeof(SmBin));

    dispatch(PH_SMVERTS, (g_nverts + VERT_CHUNK - 1) / VERT_CHUNK);
    dispatch(PH_SMSETUP, g_ntriChunks);
    dispatch(PH_SMTILES, SMT_TILES);

    /* the setup's working set is not needed again */
    for (size_t i = 0; i < (size_t)SMT_TILES * g_ntriChunks; i++) free(g_smBins[i].items);
    free(g_smBins); g_smBins = NULL;
    free(g_smTris); g_smTris = NULL;
    SDL_aligned_free(g_lsX); SDL_aligned_free(g_lsY); SDL_aligned_free(g_lsD);
}

/* ============================================================================
   Sampling: 3x3 tent PCF over 8 lanes. px/py/pz = world position, n = the
   shading (normal-mapped) normal, mv/mb = the lanes that want a result.
   Returns the lit fraction, 0..1.
   ============================================================================ */
FORCE_INLINE VF shadow_sample(VF px, VF py, VF pz, VF nx, VF ny, VF nz, VF tanT, VF mv)
{
    const VF one = _mm256_set1_ps(1.0f);
    VF noff = _mm256_set1_ps(LIT_normalOffset * LIT_normalOffsetScale);
    px = _mm256_fmadd_ps(nx, noff, px);                  /* push the sample off the surface */
    py = _mm256_fmadd_ps(ny, noff, py);
    pz = _mm256_fmadd_ps(nz, noff, pz);
    VF bias = _mm256_fmadd_ps(tanT, _mm256_set1_ps(g_smBias1), _mm256_set1_ps(g_smBias0));
    const float (*row)[4] = g_smRow;
    VF sx = _mm256_fmadd_ps(px, _mm256_set1_ps(row[0][0]), _mm256_fmadd_ps(py, _mm256_set1_ps(row[0][1]),
            _mm256_fmadd_ps(pz, _mm256_set1_ps(row[0][2]), _mm256_set1_ps(row[0][3]))));
    VF sy = _mm256_fmadd_ps(px, _mm256_set1_ps(row[1][0]), _mm256_fmadd_ps(py, _mm256_set1_ps(row[1][1]),
            _mm256_fmadd_ps(pz, _mm256_set1_ps(row[1][2]), _mm256_set1_ps(row[1][3]))));
    VF sd = _mm256_fmadd_ps(px, _mm256_set1_ps(row[2][0]), _mm256_fmadd_ps(py, _mm256_set1_ps(row[2][1]),
            _mm256_fmadd_ps(pz, _mm256_set1_ps(row[2][2]), _mm256_set1_ps(row[2][3]))));
    sd = _mm256_mul_ps(_mm256_sub_ps(sd, bias), _mm256_set1_ps((float)SM_QMAX));   /* into the map's 16-bit scale */

    VF xf = _mm256_sub_ps(sx, _mm256_set1_ps(0.5f)), yf = _mm256_sub_ps(sy, _mm256_set1_ps(0.5f));
    VF xfl = _mm256_floor_ps(xf), yfl = _mm256_floor_ps(yf);
    VF fx = _mm256_sub_ps(xf, xfl), fy = _mm256_sub_ps(yf, yfl);

    {   /* Classification: when every tap of every lane provably passes (or
           fails) against the 16x16-texel min/max cells under the footprint,
           skip the taps. Most spans settle here. */
        const VF inf = _mm256_set1_ps(1e30f), ninf = _mm256_set1_ps(-1e30f);
        float xmin = hmin8(_mm256_blendv_ps(inf, xf, mv)), xmax = hmax8(_mm256_blendv_ps(ninf, xf, mv));
        float ymin = hmin8(_mm256_blendv_ps(inf, yf, mv)), ymax = hmax8(_mm256_blendv_ps(ninf, yf, mv));
        float sdmin = hmin8(_mm256_blendv_ps(inf, sd, mv)), sdmax = hmax8(_mm256_blendv_ps(ninf, sd, mv));
#define SM_CLAMPI(v) ((int)((v) > -4.0f ? ((v) < (float)SM_RES ? (v) : (float)SM_RES) : -4.0f))   /* NaN -> -4 */
        int cx0 = SM_CLAMPI(xmin) - 2, cx1 = SM_CLAMPI(xmax) + 2;
        int cy0 = SM_CLAMPI(ymin) - 2, cy1 = SM_CLAMPI(ymax) + 2;
#undef SM_CLAMPI
        cx0 = cx0 < 0 ? 0 : cx0; cy0 = cy0 < 0 ? 0 : cy0;
        cx1 = cx1 > SM_RES - 1 ? SM_RES - 1 : cx1; cy1 = cy1 > SM_RES - 1 ? SM_RES - 1 : cy1;
        cx0 >>= SM_COARSE_SHIFT; cx1 >>= SM_COARSE_SHIFT; cy0 >>= SM_COARSE_SHIFT; cy1 >>= SM_COARSE_SHIFT;
        if ((cx1 - cx0 + 1) * (cy1 - cy0 + 1) <= 64) {   /* a huge footprint just takes the taps */
            float minD = 1e30f, maxD = -1e30f;
            for (int cy = cy0; cy <= cy1; cy++) {
                const float* pmn = g_smCoarseMin + (size_t)cy * SM_COARSE;
                const float* pmx = g_smCoarseMax + (size_t)cy * SM_COARSE;
                for (int cx = cx0; cx <= cx1; cx++) {
                    if (pmn[cx] < minD) minD = pmn[cx];
                    if (pmx[cx] > maxD) maxD = pmx[cx];
                }
            }
            if (sdmax <= minD) return one;                   /* every tap lit */
            if (sdmin > maxD) return _mm256_setzero_ps();    /* every tap shadowed */
        }
    }

    /* Penumbra: 9 taps. At 16 bits a row's three taps are 6 bytes, so ONE
       64-bit element per lane carries the whole row (tap -1 | tap 0 in the low
       dword, tap +1 in the high one). Lanes where the map border clamp broke
       that run re-fetch taps 0 and +1 on their own. */
    const VI zero = _mm256_setzero_si256(), maxc = _mm256_set1_epi32(SM_RES - 1), M16 = _mm256_set1_epi32(0xFFFF);
    const VI mask = _mm256_castps_si256(mv);
    VI xi = _mm256_cvtps_epi32(xfl), yi = _mm256_cvtps_epi32(yfl);
    VI xc[3], rc[3];
    for (int k = 0; k < 3; k++) {
        xc[k] = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(xi, _mm256_set1_epi32(k - 1)), zero), maxc);
        rc[k] = _mm256_slli_epi32(_mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(yi, _mm256_set1_epi32(k - 1)), zero), maxc), SM_SHIFT);
    }
    VI fix = _mm256_or_si256(_mm256_xor_si256(_mm256_cmpeq_epi32(xc[1], _mm256_add_epi32(xc[0], _mm256_set1_epi32(1))), mask),
                             _mm256_xor_si256(_mm256_cmpeq_epi32(xc[2], _mm256_add_epi32(xc[0], _mm256_set1_epi32(2))), mask));
    fix = _mm256_and_si256(fix, mask);
    const int fixb = mask_bitsi(fix);
    VF wx[3] = { _mm256_sub_ps(one, fx), _mm256_set1_ps(2.0f), _mm256_add_ps(one, fx) };   /* tent weights at -1, 0, +1 */
    VF wy[3] = { _mm256_sub_ps(one, fy), _mm256_set1_ps(2.0f), _mm256_add_ps(one, fy) };
    VF acc = _mm256_setzero_ps();
    for (int j = 0; j < 3; j++) {
        VI lo, hi;
        gather_pairs2(g_smDepth, _mm256_add_epi32(rc[j], xc[0]), mask, &lo, &hi);
        VI t0 = _mm256_and_si256(lo, M16), t1 = _mm256_srli_epi32(lo, 16), t2 = _mm256_and_si256(hi, M16);
        if (fixb) {
            t1 = _mm256_blendv_epi8(t1, _mm256_and_si256(_mm256_mask_i32gather_epi32(zero, (const int*)g_smDepth, _mm256_add_epi32(rc[j], xc[1]), fix, 2), M16), fix);
            t2 = _mm256_blendv_epi8(t2, _mm256_and_si256(_mm256_mask_i32gather_epi32(zero, (const int*)g_smDepth, _mm256_add_epi32(rc[j], xc[2]), fix, 2), M16), fix);
        }
        acc = _mm256_add_ps(acc, _mm256_and_ps(_mm256_mul_ps(wx[0], wy[j]), _mm256_cmp_ps(sd, _mm256_cvtepi32_ps(t0), _CMP_LE_OQ)));
        acc = _mm256_add_ps(acc, _mm256_and_ps(_mm256_mul_ps(wx[1], wy[j]), _mm256_cmp_ps(sd, _mm256_cvtepi32_ps(t1), _CMP_LE_OQ)));
        acc = _mm256_add_ps(acc, _mm256_and_ps(_mm256_mul_ps(wx[2], wy[j]), _mm256_cmp_ps(sd, _mm256_cvtepi32_ps(t2), _CMP_LE_OQ)));
    }
    return _mm256_mul_ps(acc, _mm256_set1_ps(1.0f / 16.0f));   /* the weights sum to 4 * 4 */
}
