/* geometry.c -- the first two per-frame phases.

   PH_VERTS: every vertex to clip space, frustum outcodes and screen space,
   8 at a time from the SoA positions.
   PH_TRIS: per 1024-triangle chunk, skip culled clusters, reject by outcode,
   clip against the near plane, set up integer edges and attribute planes, and
   push the triangle into the bin of every 64x64 tile its bbox touches.

   Tiled binning keeps the frame deterministic: bins are keyed by TRIANGLE
   CHUNK (never by thread), so each bin has exactly one writer and a tile's
   draw order is a pure function of triangle order. Bins clear lazily by frame
   stamp, and the first push of a frame into (chunk, tile) sets the chunk's bit
   in the tile's chunk bitmap, so a tile lists its live chunks from a few
   64-bit words instead of probing every bin header. */
#include "core.h"

/* ============================================================================
   PH_VERTS
   ============================================================================ */
/* 8 lanes of four SoA vectors -> 8 consecutive 4-float structs at dst */
FORCE_INLINE void store_aos4x8(float* dst, VF a, VF b, VF c, VF d)
{
    VF t0 = _mm256_unpacklo_ps(a, b), t1 = _mm256_unpackhi_ps(a, b);   /* a0 b0 a1 b1 | a4 b4 a5 b5 ;  a2 b2 a3 b3 | a6 .. */
    VF t2 = _mm256_unpacklo_ps(c, d), t3 = _mm256_unpackhi_ps(c, d);
    VF u0 = _mm256_shuffle_ps(t0, t2, 0x44);     /* struct 0 | 4 */
    VF u1 = _mm256_shuffle_ps(t0, t2, 0xEE);     /* struct 1 | 5 */
    VF u2 = _mm256_shuffle_ps(t1, t3, 0x44);     /* struct 2 | 6 */
    VF u3 = _mm256_shuffle_ps(t1, t3, 0xEE);     /* struct 3 | 7 */
    _mm256_store_ps(dst + 0,  _mm256_permute2f128_ps(u0, u1, 0x20));   /* structs 0, 1 */
    _mm256_store_ps(dst + 8,  _mm256_permute2f128_ps(u2, u3, 0x20));   /* structs 2, 3 */
    _mm256_store_ps(dst + 16, _mm256_permute2f128_ps(u0, u1, 0x31));   /* structs 4, 5 */
    _mm256_store_ps(dst + 24, _mm256_permute2f128_ps(u2, u3, 0x31));   /* structs 6, 7 */
}

static void job_verts(int chunk)
{
    int i0 = chunk * VERT_CHUNK, i1 = i0 + VERT_CHUNK;
    if (i1 > g_nverts) i1 = g_nverts;
    const float (*m)[4] = g_vp.m;
    VF m00 = _mm256_set1_ps(m[0][0]), m01 = _mm256_set1_ps(m[0][1]), m02 = _mm256_set1_ps(m[0][2]), m03 = _mm256_set1_ps(m[0][3]);
    VF m10 = _mm256_set1_ps(m[1][0]), m11 = _mm256_set1_ps(m[1][1]), m12 = _mm256_set1_ps(m[1][2]), m13 = _mm256_set1_ps(m[1][3]);
    VF m20 = _mm256_set1_ps(m[2][0]), m21 = _mm256_set1_ps(m[2][1]), m22 = _mm256_set1_ps(m[2][2]), m23 = _mm256_set1_ps(m[2][3]);
    VF m30 = _mm256_set1_ps(m[3][0]), m31 = _mm256_set1_ps(m[3][1]), m32 = _mm256_set1_ps(m[3][2]), m33 = _mm256_set1_ps(m[3][3]);
    VF halfW = _mm256_set1_ps(g_halfWf), halfH = _mm256_set1_ps(g_halfHf);
    VF one = _mm256_set1_ps(1.0f), zero = _mm256_setzero_ps();
    for (int i = i0; i < i1; i += 8) {
        VF x = _mm256_load_ps(g_vpx + i), y = _mm256_load_ps(g_vpy + i), z = _mm256_load_ps(g_vpz + i);
        VF cx = _mm256_fmadd_ps(m00, x, _mm256_fmadd_ps(m01, y, _mm256_fmadd_ps(m02, z, m03)));
        VF cy = _mm256_fmadd_ps(m10, x, _mm256_fmadd_ps(m11, y, _mm256_fmadd_ps(m12, z, m13)));
        VF cz = _mm256_fmadd_ps(m20, x, _mm256_fmadd_ps(m21, y, _mm256_fmadd_ps(m22, z, m23)));
        VF cw = _mm256_fmadd_ps(m30, x, _mm256_fmadd_ps(m31, y, _mm256_fmadd_ps(m32, z, m33)));
        VF ncw = _mm256_sub_ps(zero, cw);
        VI oc = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(cz, zero, _CMP_LT_OQ)), _mm256_set1_epi32(OC_NEAR));
        oc = _mm256_or_si256(oc, _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(cx, ncw, _CMP_LT_OQ)), _mm256_set1_epi32(OC_L)));
        oc = _mm256_or_si256(oc, _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(cx, cw, _CMP_GT_OQ)), _mm256_set1_epi32(OC_R)));
        oc = _mm256_or_si256(oc, _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(cy, ncw, _CMP_LT_OQ)), _mm256_set1_epi32(OC_B)));
        oc = _mm256_or_si256(oc, _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(cy, cw, _CMP_GT_OQ)), _mm256_set1_epi32(OC_T)));
        /* 8 dwords < 32 -> 8 bytes: two saturating packs, then the lanes in order */
        __m128i ocw = _mm_packus_epi32(_mm256_castsi256_si128(oc), _mm256_extracti128_si256(oc, 1));
        _mm_storel_epi64((__m128i*)(g_oc + i), _mm_packus_epi16(ocw, ocw));
        store_aos4x8((float*)(g_clip + i), cx, cy, cz, cw);
        /* screen mapping: y-down, depth cz/cw (R1). Lanes behind the eye
           produce junk nobody reads: the outcodes send them to the clipper. */
        VF iw = _mm256_div_ps(one, cw);
        VF sx = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(cx, iw), one), halfW);
        VF sy = _mm256_sub_ps(halfH, _mm256_mul_ps(_mm256_mul_ps(cy, iw), halfH));
        store_aos4x8((float*)(g_scr + i), sx, sy, _mm256_mul_ps(cz, iw), iw);
    }
}

/* ============================================================================
   PH_TRIS: setup and binning
   ============================================================================ */
/* one Tri slot; each thread claims 256 at a time so the atomic is rare */
static int alloc_tri_slot(int tid)
{
    SlotCache* s = &g_slot[tid];
    if (s->next == s->end) {
        s->next = SDL_AddAtomicInt(&g_triCount, 256);          /* returns the value before the add */
        s->end = s->next + 256;
    }
    if (s->next >= g_triCap) return -1;
    return s->next++;
}

static void bin_push(Bin* b, int v)
{
    if (b->stamp != g_frameStamp) { b->stamp = g_frameStamp; b->count = 0; }   /* first push this frame */
    if (b->count == b->cap) {
        b->cap = b->cap ? b->cap * 2 : 64;
        b->items = (int*)realloc(b->items, (size_t)b->cap * sizeof(int));
    }
    b->items[b->count++] = v;
}

FORCE_INLINE int32_t snap_subpix(float v)
{
    return _mm_cvtss_si32(_mm_set_ss(v * (float)SUBPIX_ONE));   /* nearest-even, one instruction */
}

/* Integer edge k from snapped vertex i: inside is e >= 0 and the top-left
   fill rule is folded into C as -1 before the floor shift (R3). */
#define SETUP_EDGE(t, k, Ae, Be, Xi, Yi, RX, RY) do {                                        \
        const int64_t bias_ = ((Ae) < 0) | (((Ae) == 0) & ((Be) <= 0));                        \
        (t)->C[k] = ((Ae) * ((RX) + SUBPIX_ONE / 2 - (Xi)) + (Be) * ((RY) + SUBPIX_ONE / 2 - (Yi)) - bias_) >> SUBPIX_SHIFT; \
        (t)->A[k] = (int32_t)(Ae); (t)->B[k] = (int32_t)(Be);                                  \
    } while (0)

/* the rare triangle with extents beyond EDGE_EXT_MAX (a near-clipped vertex
   far off screen): clamp the coordinates, shift oversized edges down */
static void setup_edges_wide(Tri* t, const float* xs, const float* ys, int minx, int miny)
{
    int32_t X[3], Y[3];
    for (int v = 0; v < 3; v++) {
        float sx = xs[v] * (float)SUBPIX_ONE, sy = ys[v] * (float)SUBPIX_ONE;
        sx = sx > (float)EDGE_COORD_MAX ? (float)EDGE_COORD_MAX : (sx < -(float)EDGE_COORD_MAX ? -(float)EDGE_COORD_MAX : sx);
        sy = sy > (float)EDGE_COORD_MAX ? (float)EDGE_COORD_MAX : (sy < -(float)EDGE_COORD_MAX ? -(float)EDGE_COORD_MAX : sy);
        X[v] = _mm_cvtss_si32(_mm_set_ss(sx)); Y[v] = _mm_cvtss_si32(_mm_set_ss(sy));
    }
    const int64_t RX = (int64_t)minx << SUBPIX_SHIFT, RY = (int64_t)miny << SUBPIX_SHIFT;
    for (int k = 0; k < 3; k++) {
        const int i = k == 2 ? 0 : k + 1, j = k == 0 ? 2 : k - 1;
        int64_t A = (int64_t)Y[j] - Y[i], B = (int64_t)X[i] - X[j];
        int64_t mag = (A < 0 ? -A : A) + (B < 0 ? -B : B);
        if (mag > EDGE_MAG_MAX) {
            int sh = 0;
            do { mag >>= 1; sh++; } while (mag > EDGE_MAG_MAX);
            A /= (int64_t)1 << sh; B /= (int64_t)1 << sh;
        }
        SETUP_EDGE(t, k, A, B, X[i], Y[i], RX, RY);
    }
}

static void setup_and_bin(const ScrVert* s0, const ScrVert* s1, const ScrVert* s2,
                          const Vertex* a0, const Vertex* a1, const Vertex* a2, int mat, int src, int tid)
{
    float x0 = s0->sx, y0 = s0->sy, x1 = s1->sx, y1 = s1->sy, x2 = s2->sx, y2 = s2->sy;
    float area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
    int isBack = 0;
    if (area2 > -1e-6f && area2 < 1e-6f) return;           /* degenerate */
    if (area2 > 0) {                                        /* back face: fronts wind negative (R3) */
        if (g_triCullBack[src]) return;                     /* single-sided material */
        const ScrVert* ts = s1; s1 = s2; s2 = ts;           /* double-sided: reorient, remember */
        const Vertex* ta = a1; a1 = a2; a2 = ta;
        x1 = s1->sx; y1 = s1->sy; x2 = s2->sx; y2 = s2->sy;
        isBack = 1;
    }
    float mnx = x0 < x1 ? (x0 < x2 ? x0 : x2) : (x1 < x2 ? x1 : x2);
    float mxx = x0 > x1 ? (x0 > x2 ? x0 : x2) : (x1 > x2 ? x1 : x2);
    float mny = y0 < y1 ? (y0 < y2 ? y0 : y2) : (y1 < y2 ? y1 : y2);
    float mxy = y0 > y1 ? (y0 > y2 ? y0 : y2) : (y1 > y2 ? y1 : y2);
    if (mxx < 0 || mxy < 0 || mnx >= g_width || mny >= g_height) return;   /* off screen */
    int minx = (int)mnx; if (minx < 0) minx = 0;
    int maxx = (int)mxx; if (maxx > g_width - 1) maxx = g_width - 1;
    int miny = (int)mny; if (miny < 0) miny = 0;
    int maxy = (int)mxy; if (maxy > g_height - 1) maxy = g_height - 1;
    if (minx > maxx || miny > maxy) return;

    int slot = alloc_tri_slot(tid);
    if (slot < 0) return;
    Tri* t = &g_tris[slot];

    /* ---- fixed-point edges ---- */
    if ((mxx - mnx) + (mxy - mny) < (float)EDGE_EXT_MAX) {
        const int32_t X0 = snap_subpix(x0), Y0 = snap_subpix(y0);
        const int32_t X1 = snap_subpix(x1), Y1 = snap_subpix(y1);
        const int32_t X2 = snap_subpix(x2), Y2 = snap_subpix(y2);
        const int64_t RX = (int64_t)minx << SUBPIX_SHIFT, RY = (int64_t)miny << SUBPIX_SHIFT;
        const int64_t A0 = (int64_t)Y2 - Y1, B0 = (int64_t)X1 - X2;
        const int64_t A1 = (int64_t)Y0 - Y2, B1 = (int64_t)X2 - X0;
        const int64_t A2 = (int64_t)Y1 - Y0, B2 = (int64_t)X0 - X1;
        SETUP_EDGE(t, 0, A0, B0, X1, Y1, RX, RY);
        SETUP_EDGE(t, 1, A1, B1, X2, Y2, RX, RY);
        SETUP_EDGE(t, 2, A2, B2, X0, Y0, RX, RY);
    } else {
        const float xs[3] = { x0, x1, x2 }, ys[3] = { y0, y1, y2 };
        setup_edges_wide(t, xs, ys, minx, miny);
    }

    /* ---- attribute planes, in double at the local origin (minx, miny) ----
       Solves value(x, y) = c + gx*(x - minx) + gy*(y - miny) through the three
       vertex values by Cramer's rule; invA = 1/(2*area) taken once. */
    const double x0d = x0, y0d = y0;
    const double dx10 = (double)x1 - x0d, dy10 = (double)y1 - y0d, dx20 = (double)x2 - x0d, dy20 = (double)y2 - y0d;
    const double invA = _mm_cvtsd_f64(_mm_div_sd(_mm_set_sd(1.0), _mm_set_sd(dx10 * dy20 - dy10 * dx20)));
    const double ox = (double)minx - x0d, oy = (double)miny - y0d;
#define PLANE3(a0v, a1v, a2v, cOut, gxOut, gyOut) do {                                   \
        const double d1_ = (double)(a1v) - (a0v), d2_ = (double)(a2v) - (a0v);            \
        const double gx_ = (d1_ * dy20 - d2_ * dy10) * invA, gy_ = (d2_ * dx10 - d1_ * dx20) * invA; \
        (cOut) = (float)((a0v) + gx_ * ox + gy_ * oy); (gxOut) = (float)gx_; (gyOut) = (float)gy_; \
    } while (0)
    PLANE3(s0->z, s1->z, s2->z, t->zc, t->zgx, t->zgy);
    if (isBack) t->zc += 1e-4f;                             /* back faces lose coplanar ties */
    {
        float z0v = s0->z, z1v = s1->z, z2v = s2->z;
        t->minz = (z0v < z1v ? (z0v < z2v ? z0v : z2v) : (z1v < z2v ? z1v : z2v)) - 1e-5f;
    }
    const float iw0 = s0->invW, iw1 = s1->invW, iw2 = s2->invW;
    PLANE3(iw0, iw1, iw2, t->wc, t->wgx, t->wgy);
    {   /* the eight perspective-correct planes at once: Vertex is eight floats,
           premultiplied by 1/w, widened to two 4-lane double vectors */
        const __m256d vdy20 = _mm256_set1_pd(dy20), vdy10 = _mm256_set1_pd(dy10);
        const __m256d vdx10 = _mm256_set1_pd(dx10), vdx20 = _mm256_set1_pd(dx20);
        const __m256d vinvA = _mm256_set1_pd(invA), vox = _mm256_set1_pd(ox), voy = _mm256_set1_pd(oy);
        SDL_ALIGNED(32) float fc[8], fgx[8], fgy[8];
        VF q0 = _mm256_mul_ps(_mm256_loadu_ps(&a0->px), _mm256_set1_ps(iw0));
        VF q1 = _mm256_mul_ps(_mm256_loadu_ps(&a1->px), _mm256_set1_ps(iw1));
        VF q2 = _mm256_mul_ps(_mm256_loadu_ps(&a2->px), _mm256_set1_ps(iw2));
        for (int h = 0; h < 2; h++) {                       /* lanes 0-3: px py pz u, lanes 4-7: v nx ny nz */
            __m256d p0 = _mm256_cvtps_pd(h ? _mm256_extractf128_ps(q0, 1) : _mm256_castps256_ps128(q0));
            __m256d p1 = _mm256_cvtps_pd(h ? _mm256_extractf128_ps(q1, 1) : _mm256_castps256_ps128(q1));
            __m256d p2 = _mm256_cvtps_pd(h ? _mm256_extractf128_ps(q2, 1) : _mm256_castps256_ps128(q2));
            __m256d d1 = _mm256_sub_pd(p1, p0), d2 = _mm256_sub_pd(p2, p0);
            __m256d gx = _mm256_mul_pd(_mm256_fmsub_pd(d1, vdy20, _mm256_mul_pd(d2, vdy10)), vinvA);
            __m256d gy = _mm256_mul_pd(_mm256_fmsub_pd(d2, vdx10, _mm256_mul_pd(d1, vdx20)), vinvA);
            __m256d cc = _mm256_fmadd_pd(gy, voy, _mm256_fmadd_pd(gx, vox, p0));
            _mm_store_ps(fc + 4 * h, _mm256_cvtpd_ps(cc));
            _mm_store_ps(fgx + 4 * h, _mm256_cvtpd_ps(gx));
            _mm_store_ps(fgy + 4 * h, _mm256_cvtpd_ps(gy));
        }
        const float ns = isBack ? -1.0f : 1.0f;             /* back faces: normals mirrored per triangle (R2) */
        t->pc[0] = fc[0]; t->pc[1] = fc[1]; t->pc[2] = fc[2]; t->uc = fc[3]; t->vc = fc[4];
        t->nc[0] = fc[5] * ns; t->nc[1] = fc[6] * ns; t->nc[2] = fc[7] * ns;
        t->pgx[0] = fgx[0]; t->pgx[1] = fgx[1]; t->pgx[2] = fgx[2]; t->ugx = fgx[3]; t->vgx = fgx[4];
        t->ngx[0] = fgx[5] * ns; t->ngx[1] = fgx[6] * ns; t->ngx[2] = fgx[7] * ns;
        t->pgy[0] = fgy[0]; t->pgy[1] = fgy[1]; t->pgy[2] = fgy[2]; t->ugy = fgy[3]; t->vgy = fgy[4];
        t->ngy[0] = fgy[5] * ns; t->ngy[1] = fgy[6] * ns; t->ngy[2] = fgy[7] * ns;
    }
#undef PLANE3
    {   /* face tangent for the normal map (R6): T = normalize(e1*dv2 - e2*dv1) * sign(det) */
        float e1x = a1->px - a0->px, e1y = a1->py - a0->py, e1z = a1->pz - a0->pz;
        float e2x = a2->px - a0->px, e2y = a2->py - a0->py, e2z = a2->pz - a0->pz;
        float du1 = a1->u - a0->u, dv1 = a1->v - a0->v, du2 = a2->u - a0->u, dv2 = a2->v - a0->v;
        float det = du1 * dv2 - du2 * dv1;
        float tx = e1x * dv2 - e2x * dv1, ty = e1y * dv2 - e2y * dv1, tz = e1z * dv2 - e2z * dv1;
        float l2 = tx * tx + ty * ty + tz * tz;
        if (det != 0.0f && l2 > 1e-24f) {
            float sg = det > 0.0f ? 1.0f : -1.0f;
            float sc = (isBack ? -sg : sg) * (32767.0f / sqrtf(l2));   /* back faces negate T with the normal */
            t->tq[0] = (int16_t)_mm_cvtss_si32(_mm_set_ss(tx * sc));
            t->tq[1] = (int16_t)_mm_cvtss_si32(_mm_set_ss(ty * sc));
            t->tq[2] = (int16_t)_mm_cvtss_si32(_mm_set_ss(tz * sc));
            t->ts = (int16_t)sg;
        } else { t->tq[0] = t->tq[1] = t->tq[2] = 0; t->ts = 1; }   /* degenerate uvs: the smooth normal survives */
    }
    t->minx = minx; t->maxx = maxx; t->miny = miny; t->maxy = maxy;
    t->mat = mat; t->src = src;
    t->masked = g_mats[mat].alphaMask;

    /* ---- binning ---- */
    const int ch = src >> TRI_CHUNK_SHIFT;
    for (int ty = miny >> TILE_SHIFT; ty <= (maxy >> TILE_SHIFT); ty++)
        for (int tx = minx >> TILE_SHIFT; tx <= (maxx >> TILE_SHIFT); tx++) {
            const int tile = ty * g_tilesX + tx;
            Bin* b = &g_bins[(size_t)ch * g_ntiles + tile];
            if (b->stamp != g_frameStamp) {                 /* first push into (chunk, tile): the tile's chunk bit */
                SDL_AtomicInt* word = &g_tileChunkBits[(size_t)tile * g_chunkWords + (ch >> 5)];
                const int bit = (int)(1u << (ch & 31));
                for (;;) {                                  /* atomic OR: other chunks' jobs set bits in the same word */
                    const int old = SDL_GetAtomicInt(word);
                    if ((old & bit) || SDL_CompareAndSwapAtomicInt(word, old, old | bit)) break;
                }
            }
            bin_push(b, slot);
        }
}

/* Sutherland-Hodgman against the near plane (cz >= 0), then a fan (R1) */
static void clip_and_setup(int tri, int tid)
{
    const uint32_t i0 = g_idx[tri * 3], i1 = g_idx[tri * 3 + 1], i2 = g_idx[tri * 3 + 2];
    V4 inPos[3] = { g_clip[i0], g_clip[i1], g_clip[i2] };
    Vertex inAttr[3] = { g_verts[i0], g_verts[i1], g_verts[i2] };
    V4 outPos[4]; Vertex outAttr[4]; int nOut = 0;
    for (int i = 0; i < 3; i++) {
        const V4 pc = inPos[i], pn = inPos[(i + 1) % 3];
        const Vertex ac = inAttr[i], an = inAttr[(i + 1) % 3];
        const int cIn = pc.z >= 0, nIn = pn.z >= 0;
        if (cIn) { outPos[nOut] = pc; outAttr[nOut] = ac; nOut++; }
        if (cIn != nIn) {
            const float t = pc.z / (pc.z - pn.z);
            V4* op = &outPos[nOut]; Vertex* oa = &outAttr[nOut];
            op->x = pc.x + (pn.x - pc.x) * t; op->y = pc.y + (pn.y - pc.y) * t;
            op->z = 0.0f;                    op->w = pc.w + (pn.w - pc.w) * t;
            oa->px = ac.px + (an.px - ac.px) * t; oa->py = ac.py + (an.py - ac.py) * t; oa->pz = ac.pz + (an.pz - ac.pz) * t;
            oa->u = ac.u + (an.u - ac.u) * t;     oa->v = ac.v + (an.v - ac.v) * t;
            oa->nx = ac.nx + (an.nx - ac.nx) * t; oa->ny = ac.ny + (an.ny - ac.ny) * t; oa->nz = ac.nz + (an.nz - ac.nz) * t;
            nOut++;
        }
    }
    if (nOut < 3) return;
    ScrVert sv[4];
    for (int i = 0; i < nOut; i++) {
        const float invW = 1.0f / outPos[i].w;
        sv[i].sx = (outPos[i].x * invW + 1.0f) * g_halfWf;
        sv[i].sy = g_halfHf - outPos[i].y * invW * g_halfHf;
        sv[i].z = outPos[i].z * invW;
        sv[i].invW = invW;
    }
    for (int k = 2; k < nOut; k++)
        setup_and_bin(&sv[0], &sv[k - 1], &sv[k], &outAttr[0], &outAttr[k - 1], &outAttr[k], g_triMat[tri], tri, tid);
}

static void job_tris(int chunk, int tid)
{
    int t0 = chunk << TRI_CHUNK_SHIFT, t1 = t0 + TRI_CHUNK;
    if (t1 > g_ntris) t1 = g_ntris;
    for (int c0 = t0; c0 < t1; c0 += CLUSTER_SIZE) {
        if (g_cluCull[c0 >> CLUSTER_SHIFT]) continue;      /* the whole cluster is provably invisible */
        const int c1 = c0 + CLUSTER_SIZE < t1 ? c0 + CLUSTER_SIZE : t1;
        for (int t = c0; t < c1; t++) {
            const uint32_t i0 = g_idx[t * 3], i1 = g_idx[t * 3 + 1], i2 = g_idx[t * 3 + 2];
            const uint8_t oc0 = g_oc[i0], oc1 = g_oc[i1], oc2 = g_oc[i2];
            if (oc0 & oc1 & oc2) continue;                  /* all three outside one plane */
            if ((oc0 | oc1 | oc2) & OC_NEAR) { clip_and_setup(t, tid); continue; }
            setup_and_bin(&g_scr[i0], &g_scr[i1], &g_scr[i2], &g_verts[i0], &g_verts[i1], &g_verts[i2], g_triMat[t], t, tid);
        }
    }
}

/* ============================================================================
   Cluster culling: 64 consecutive triangles share a bounding sphere and a
   face-normal cone (built once, the scene is static). Per frame, a cluster
   whose sphere lies fully outside a side or near plane, or whose whole cone
   faces away (single-sided clusters only), is skipped in PH_TRIS before any
   of its indices are loaded. Both tests carry slack, so they only remove
   triangles the per-triangle tests would reject anyway.
   ============================================================================ */
static void clusters_build(void)
{
    g_nclusters = (g_ntris + CLUSTER_SIZE - 1) >> CLUSTER_SHIFT;
    g_clusters = (Cluster*)calloc((size_t)g_nclusters, sizeof(Cluster));
    g_cluCull = (uint8_t*)calloc((size_t)g_nclusters, 1);
    for (int c = 0; c < g_nclusters; c++) {
        int t0 = c << CLUSTER_SHIFT, t1 = t0 + CLUSTER_SIZE;
        if (t1 > g_ntris) t1 = g_ntris;
        V3 mn = v3(1e30f, 1e30f, 1e30f), mx = v3(-1e30f, -1e30f, -1e30f), axis = v3(0, 0, 0);
        int allCull = 1, valid = 1;
        for (int t = t0; t < t1; t++) {
            if (!g_triCullBack[t]) allCull = 0;
            const Vertex* vv[3] = { &g_verts[g_idx[t * 3]], &g_verts[g_idx[t * 3 + 1]], &g_verts[g_idx[t * 3 + 2]] };
            for (int k = 0; k < 3; k++) {
                if (vv[k]->px < mn.x) mn.x = vv[k]->px;
                if (vv[k]->py < mn.y) mn.y = vv[k]->py;
                if (vv[k]->pz < mn.z) mn.z = vv[k]->pz;
                if (vv[k]->px > mx.x) mx.x = vv[k]->px;
                if (vv[k]->py > mx.y) mx.y = vv[k]->py;
                if (vv[k]->pz > mx.z) mx.z = vv[k]->pz;
            }
            V3 e1 = v3(vv[1]->px - vv[0]->px, vv[1]->py - vv[0]->py, vv[1]->pz - vv[0]->pz);
            V3 e2 = v3(vv[2]->px - vv[0]->px, vv[2]->py - vv[0]->py, vv[2]->pz - vv[0]->pz);
            V3 fn = v3_cross(e2, e1);                       /* the engine's visible side */
            float l2 = v3_dot(fn, fn);
            if (l2 < 1e-16f) { valid = 0; continue; }
            axis = v3_add(axis, v3_scale(fn, 1.0f / sqrtf(l2)));
        }
        Cluster* cl = &g_clusters[c];
        cl->c = v3_scale(v3_add(mn, mx), 0.5f);
        V3 he = v3_scale(v3_sub(mx, mn), 0.5f);
        cl->r = sqrtf(v3_dot(he, he)) + 1e-4f;
        float al = sqrtf(v3_dot(axis, axis));
        if (al < 1e-6f) valid = 0;
        if (valid) {
            V3 a = v3_scale(axis, 1.0f / al);
            float ct = 1.0f;
            for (int t = t0; t < t1; t++) {                 /* the cone's half-angle: the worst triangle */
                const Vertex* p0 = &g_verts[g_idx[t * 3]], *p1 = &g_verts[g_idx[t * 3 + 1]], *p2 = &g_verts[g_idx[t * 3 + 2]];
                V3 e1 = v3(p1->px - p0->px, p1->py - p0->py, p1->pz - p0->pz);
                V3 e2 = v3(p2->px - p0->px, p2->py - p0->py, p2->pz - p0->pz);
                float d = v3_dot(a, v3_norm(v3_cross(e2, e1)));
                if (d < ct) ct = d;
            }
            ct -= 1e-4f;
            if (ct <= 0.05f) valid = 0;
            else { cl->axis = a; cl->cosT = ct; cl->sinT = sqrtf(1.0f - ct * ct); }
        }
        cl->ok = (uint8_t)((valid ? 1 : 0) | (allCull ? 2 : 0));
    }
}

static void clusters_classify(void)
{
    float pl[5][4];                                          /* Gribb-Hartmann planes of the outcode tests */
    for (int k = 0; k < 4; k++) {
        pl[0][k] = g_vp.m[3][k] + g_vp.m[0][k];
        pl[1][k] = g_vp.m[3][k] - g_vp.m[0][k];
        pl[2][k] = g_vp.m[3][k] + g_vp.m[1][k];
        pl[3][k] = g_vp.m[3][k] - g_vp.m[1][k];
        pl[4][k] = g_vp.m[2][k];
    }
    for (int p = 0; p < 5; p++) {
        float l = sqrtf(pl[p][0] * pl[p][0] + pl[p][1] * pl[p][1] + pl[p][2] * pl[p][2]);
        if (l > 1e-20f) for (int k = 0; k < 4; k++) pl[p][k] /= l;
    }
    const V3 E = g_camPos;
    for (int c = 0; c < g_nclusters; c++) {
        const Cluster* cl = &g_clusters[c];
        uint8_t cull = 0;
        for (int p = 0; p < 5 && !cull; p++)
            if (pl[p][0] * cl->c.x + pl[p][1] * cl->c.y + pl[p][2] * cl->c.z + pl[p][3] < -(cl->r + 1e-3f)) cull = 1;
        if (!cull && cl->ok == 3) {
            float vx = E.x - cl->c.x, vy = E.y - cl->c.y, vz = E.z - cl->c.z;
            float d2 = vx * vx + vy * vy + vz * vz, rp = cl->r + 1e-3f;
            if (d2 > rp * rp) {                              /* eye outside the sphere */
                float dist = sqrtf(d2);
                float ca = (vx * cl->axis.x + vy * cl->axis.y + vz * cl->axis.z) / dist;
                if (ca < 0.0f) {
                    if (ca < -1.0f) ca = -1.0f;
                    float sa = sqrtf(1.0f - ca * ca);
                    /* the most eye-facing normal in the cone still faces away from every point of the sphere */
                    if (dist * (ca * cl->cosT + sa * cl->sinT) + cl->r < -(1e-4f * dist + 1e-5f)) cull = 1;
                }
            }
        }
        g_cluCull[c] = cull;
    }
}
