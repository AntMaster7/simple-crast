/* raster.c -- the third per-frame phase, PH_TILES: one 64x64 tile end to end.

   Per tile: clear its depth, run a depth-only prepass over the opaque
   triangles, run the shade pass (less-or-equal against the prepass depth for
   opaques; less-than with a depth write for alpha-tested cut-outs), shade the
   sky where no depth was written, and stream the finished tile out.

   Everything is shaded into the tile's OWN 64x64 colour buffer on the stack
   (L1/L2-resident beside the tile depth); the frame buffer is touched once,
   at the end, in whole streamed rows.

   The raster advances one 4x2 block (8 pixels = 8 AVX2 lanes) at a time. */
#include "core.h"

static const SDL_ALIGNED(32) float   SPAN_LX[8]  = { 0, 1, 2, 3, 0, 1, 2, 3 };
static const SDL_ALIGNED(32) float   SPAN_LY[8]  = { 0, 0, 0, 0, 1, 1, 1, 1 };
static const SDL_ALIGNED(32) int32_t SPAN_LXI[8] = { 0, 1, 2, 3, 0, 1, 2, 3 };
static const SDL_ALIGNED(32) int32_t SPAN_LYI[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };
#define SPAN_XREP(xb)    ((unsigned)(xb) * 0x11u)          /* a 4-bit column run in both block rows */
static const unsigned char SPAN_YEXP[4] = { 0x00, 0x0F, 0xF0, 0xFF };   /* a 2-bit row run widened to lanes */
#define SPAN_RUN(lo, hi) ((2u << (hi)) - (1u << (lo)))       /* bits lo..hi */

/* Store a block's 8 colours: row 0 = lanes 0-3 at dst, row 1 = lanes 4-7 one
   tile row below. The tile buffer is thread-private, so a masked store is a
   plain load, blend and store. */
FORCE_INLINE void span_store(uint32_t* dst, VI mask, VI c)
{
    __m128i m0 = _mm256_castsi256_si128(mask), m1 = _mm256_extracti128_si256(mask, 1);
    __m128i r0 = _mm_loadu_si128((const __m128i*)dst), r1 = _mm_loadu_si128((const __m128i*)(dst + TILE_SIZE));
    _mm_storeu_si128((__m128i*)dst, _mm_blendv_epi8(r0, _mm256_castsi256_si128(c), m0));
    _mm_storeu_si128((__m128i*)(dst + TILE_SIZE), _mm_blendv_epi8(r1, _mm256_extracti128_si256(c, 1), m1));
}

/* Reinhard per channel, gamma-2 encode, round to bytes, pack BGRA (R15) */
FORCE_INLINE VI pack_color(VF r, VF g, VF b)
{
    const VF one = _mm256_set1_ps(1.0f), s255 = _mm256_set1_ps(255.0f);
    r = _mm256_sqrt_ps(_mm256_mul_ps(r, v_rcp(_mm256_add_ps(r, one))));
    g = _mm256_sqrt_ps(_mm256_mul_ps(g, v_rcp(_mm256_add_ps(g, one))));
    b = _mm256_sqrt_ps(_mm256_mul_ps(b, v_rcp(_mm256_add_ps(b, one))));
    VI ri = _mm256_cvtps_epi32(_mm256_mul_ps(r, s255));
    VI gi = _mm256_cvtps_epi32(_mm256_mul_ps(g, s255));
    VI bi = _mm256_cvtps_epi32(_mm256_mul_ps(b, s255));
    return _mm256_or_si256(_mm256_set1_epi32((int)0xFF000000),
           _mm256_or_si256(_mm256_slli_epi32(ri, 16), _mm256_or_si256(_mm256_slli_epi32(gi, 8), bi)));
}

/* ============================================================================
   Bilinear texture fetch (R4): per-lane mip level, repeat wrap, texel pairs
   gathered as 64-bit elements, the blend in 16-bit integer fields.
   ============================================================================ */
typedef struct TexCoords {
    VI idx0, idx1;                    /* top / bottom row: element of the left texel of each pair */
    VI idx0w, idx1w;                  /* the right texel alone, for lanes whose u wrapped */
    VI fx, fy;                        /* lerp weights, 0..256 */
    VI xw; int xwb;                   /* the wrapped lanes */
} TexCoords;

FORCE_INLINE void tex_coords(const Texture* tx, VF u, VF v, VI mip, VI mask, int mb, TexCoords* tc)
{
    const VI zero = _mm256_setzero_si256(), onei = _mm256_set1_epi32(1);
    mip = _mm256_min_epi32(_mm256_max_epi32(mip, zero), _mm256_set1_epi32(tx->nmips - 1));
    VI iw, ih, off;
    {   /* The per-level constants. Almost always every lane of a block sits on
           ONE level: broadcast it. Otherwise gather from the level tables. */
        SDL_ALIGNED(32) int32_t lv[8];
        _mm256_store_si256((__m256i*)lv, mip);
        const unsigned first = ctz32((uint32_t)mb);
        const int L = lv[first];
        if ((mask_bitsi(_mm256_cmpeq_epi32(mip, _mm256_set1_epi32(L))) & mb) == mb) {
            iw = _mm256_set1_epi32(tx->mw[L]); ih = _mm256_set1_epi32(tx->mh[L]); off = _mm256_set1_epi32(tx->moff[L]);
        } else {
            iw = _mm256_i32gather_epi32(tx->mw, mip, 4);
            ih = _mm256_i32gather_epi32(tx->mh, mip, 4);
            off = _mm256_i32gather_epi32(tx->moff, mip, 4);
        }
    }
    const VI iwm1 = _mm256_sub_epi32(iw, onei), ihm1 = _mm256_sub_epi32(ih, onei);
    const VF half = _mm256_set1_ps(0.5f), s256 = _mm256_set1_ps(256.0f);
    u = _mm256_sub_ps(u, _mm256_floor_ps(u));                /* repeat */
    v = _mm256_sub_ps(v, _mm256_floor_ps(v));
    VF xf = _mm256_fmsub_ps(u, _mm256_cvtepi32_ps(iw), half);  /* texel-centre addressing */
    VF yf = _mm256_fmsub_ps(v, _mm256_cvtepi32_ps(ih), half);
    VF xfl = _mm256_floor_ps(xf), yfl = _mm256_floor_ps(yf);
    VI fx = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sub_ps(xf, xfl), s256));
    VI fy = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sub_ps(yf, yfl), s256));
    const VI c256 = _mm256_set1_epi32(256);
    tc->fx = _mm256_min_epi32(_mm256_max_epi32(fx, zero), c256);
    tc->fy = _mm256_min_epi32(_mm256_max_epi32(fy, zero), c256);
    VI x0 = _mm256_cvtps_epi32(xfl), y0 = _mm256_cvtps_epi32(yfl);
    /* -1 (u just under half a texel) wraps to the last column; junk clamps into range */
    x0 = _mm256_blendv_epi8(_mm256_min_epi32(x0, iwm1), iwm1, _mm256_cmpgt_epi32(zero, x0));
    y0 = _mm256_blendv_epi8(_mm256_min_epi32(y0, ihm1), ihm1, _mm256_cmpgt_epi32(zero, y0));
    x0 = _mm256_max_epi32(x0, zero); y0 = _mm256_max_epi32(y0, zero);
    VI x1 = _mm256_add_epi32(x0, onei), y1 = _mm256_add_epi32(y0, onei);
    tc->xw = _mm256_and_si256(_mm256_cmpgt_epi32(x1, iwm1), mask);   /* right texel wrapped to column 0 */
    tc->xwb = mask_bitsi(tc->xw);
    x1 = _mm256_andnot_si256(tc->xw, x1);
    y1 = _mm256_andnot_si256(_mm256_cmpgt_epi32(y1, ihm1), y1);
    const VI row0 = _mm256_add_epi32(off, _mm256_mullo_epi32(y0, iw)), row1 = _mm256_add_epi32(off, _mm256_mullo_epi32(y1, iw));
    tc->idx0 = _mm256_add_epi32(row0, x0); tc->idx1 = _mm256_add_epi32(row1, x0);
    tc->idx0w = _mm256_add_epi32(row0, x1); tc->idx1w = _mm256_add_epi32(row1, x1);
}

/* the gathers and the blend: lfin = B | R<<16, hfin = G | A<<16 (fields 0..255) */
FORCE_INLINE void tex_fetch(const uint32_t* texels, const TexCoords* tc, VI mask, VI* lfin, VI* hfin)
{
    VI t00, t01, t10, t11;
    gather_pairs4(texels, tc->idx0, mask, &t00, &t01);
    gather_pairs4(texels, tc->idx1, mask, &t10, &t11);
    if (tc->xwb) {                                           /* rare: re-fetch the wrapped right texels */
        t01 = _mm256_mask_i32gather_epi32(t01, (const int*)texels, tc->idx0w, tc->xw, 4);
        t11 = _mm256_mask_i32gather_epi32(t11, (const int*)texels, tc->idx1w, tc->xw, 4);
    }
    /* A texel splits into two dwords of two 16-bit fields; a dword multiply by
       an 8-bit weight scales both fields without carries, and one word shift
       takes the weight back out of both. */
    const VI M16 = _mm256_set1_epi32(0x00FF00FF), c256 = _mm256_set1_epi32(256);
    const VI w1x = tc->fx, w0x = _mm256_sub_epi32(c256, tc->fx), w1y = tc->fy, w0y = _mm256_sub_epi32(c256, tc->fy);
    VI l00 = _mm256_and_si256(t00, M16), h00 = _mm256_srli_epi16(t00, 8);
    VI l01 = _mm256_and_si256(t01, M16), h01 = _mm256_srli_epi16(t01, 8);
    VI l10 = _mm256_and_si256(t10, M16), h10 = _mm256_srli_epi16(t10, 8);
    VI l11 = _mm256_and_si256(t11, M16), h11 = _mm256_srli_epi16(t11, 8);
#define LERP16(s0, s1, w0, w1) _mm256_srli_epi16(_mm256_add_epi32(_mm256_mullo_epi32(s0, w0), _mm256_mullo_epi32(s1, w1)), 8)
    VI lt = LERP16(l00, l01, w0x, w1x), ht = LERP16(h00, h01, w0x, w1x);
    VI lb = LERP16(l10, l11, w0x, w1x), hb = LERP16(h10, h11, w0x, w1x);
    *lfin = LERP16(lt, lb, w0y, w1y);
    *hfin = LERP16(ht, hb, w0y, w1y);
#undef LERP16
}

/* ============================================================================
   shade8: shade the covered lanes of one triangle's 4x2 block and store them.
   X/Y are the lanes' pixel centres in the triangle's local frame. Returns the
   lanes that survived the alpha test.
   ============================================================================ */
#define SH_PLANE(T, gx, gy, c) _mm256_fmadd_ps(_mm256_set1_ps((T)->gx), X, _mm256_fmadd_ps(_mm256_set1_ps((T)->gy), Y, _mm256_set1_ps((T)->c)))

FORCE_INLINE int shade8(const Tri* t, VI mask, int mb, VF X, VF Y, uint32_t* dst, int alphaTest)
{
    const Material* mat = &g_mats[t->mat];
    const VF one = _mm256_set1_ps(1.0f), zero = _mm256_setzero_ps(), eps = _mm256_set1_ps(1e-12f);

    /* ---- perspective-correct attributes (R3) ---- */
    VF w = v_rcp(SH_PLANE(t, wgx, wgy, wc));
    VF u = _mm256_mul_ps(SH_PLANE(t, ugx, ugy, uc), w);
    VF v = _mm256_mul_ps(SH_PLANE(t, vgx, vgy, vc), w);
    VF nx = SH_PLANE(t, ngx[0], ngy[0], nc[0]), ny = SH_PLANE(t, ngx[1], ngy[1], nc[1]), nz = SH_PLANE(t, ngx[2], ngy[2], nc[2]);
    VF nl = v_rsqrt(_mm256_max_ps(_mm256_fmadd_ps(nx, nx, _mm256_fmadd_ps(ny, ny, _mm256_mul_ps(nz, nz))), eps));
    nx = _mm256_mul_ps(nx, nl); ny = _mm256_mul_ps(ny, nl); nz = _mm256_mul_ps(nz, nl);

    /* ---- the mip level (R4): analytic derivatives by the quotient rule,
       level = floor(0.5*log2(rho2) + 0.5) = exponent(2*rho2) >> 1 ---- */
    const VF wgx = _mm256_set1_ps(t->wgx), wgy = _mm256_set1_ps(t->wgy);
    const VF tw = _mm256_set1_ps((float)mat->tex->w0), th = _mm256_set1_ps((float)mat->tex->h0);
    VF dudx = _mm256_mul_ps(_mm256_mul_ps(w, _mm256_fnmadd_ps(u, wgx, _mm256_set1_ps(t->ugx))), tw);
    VF dvdx = _mm256_mul_ps(_mm256_mul_ps(w, _mm256_fnmadd_ps(v, wgx, _mm256_set1_ps(t->vgx))), th);
    VF dudy = _mm256_mul_ps(_mm256_mul_ps(w, _mm256_fnmadd_ps(u, wgy, _mm256_set1_ps(t->ugy))), tw);
    VF dvdy = _mm256_mul_ps(_mm256_mul_ps(w, _mm256_fnmadd_ps(v, wgy, _mm256_set1_ps(t->vgy))), th);
    VF rho2 = _mm256_max_ps(_mm256_fmadd_ps(dudx, dudx, _mm256_mul_ps(dvdx, dvdx)), _mm256_fmadd_ps(dudy, dudy, _mm256_mul_ps(dvdy, dvdy)));
    VI mip = _mm256_srai_epi32(v_exponent(_mm256_mul_ps(_mm256_max_ps(rho2, eps), _mm256_set1_ps(2.0f))), 1);

    /* ---- albedo, and the normal map (R6) ---- */
    TexCoords tc; tex_coords(mat->tex, u, v, mip, mask, mb, &tc);
    VI lfin, hfin;
    if (mat->nrm) {
        const float tqs = 1.0f / 32767.0f;
        VF tx = _mm256_set1_ps(t->tq[0] * tqs), ty = _mm256_set1_ps(t->tq[1] * tqs), tz = _mm256_set1_ps(t->tq[2] * tqs);
        const VF ts = _mm256_set1_ps((float)t->ts);
        VF tdn = _mm256_fmadd_ps(tx, nx, _mm256_fmadd_ps(ty, ny, _mm256_mul_ps(tz, nz)));   /* Gram-Schmidt against N */
        tx = _mm256_fnmadd_ps(nx, tdn, tx); ty = _mm256_fnmadd_ps(ny, tdn, ty); tz = _mm256_fnmadd_ps(nz, tdn, tz);
        VF tl = v_rsqrt(_mm256_max_ps(_mm256_fmadd_ps(tx, tx, _mm256_fmadd_ps(ty, ty, _mm256_mul_ps(tz, tz))), eps));
        tx = _mm256_mul_ps(tx, tl); ty = _mm256_mul_ps(ty, tl); tz = _mm256_mul_ps(tz, tl);
        VF bx = _mm256_mul_ps(ts, _mm256_fmsub_ps(ny, tz, _mm256_mul_ps(nz, ty)));          /* B = ts * cross(N, T') */
        VF by = _mm256_mul_ps(ts, _mm256_fmsub_ps(nz, tx, _mm256_mul_ps(nx, tz)));
        VF bz = _mm256_mul_ps(ts, _mm256_fmsub_ps(nx, ty, _mm256_mul_ps(ny, tx)));
        tex_fetch(mat->nrm->texels, &tc, mask, &lfin, &hfin);   /* same size as the base colour: same coordinates */
        const VF two255 = _mm256_set1_ps(2.0f / 255.0f), m1 = _mm256_set1_ps(-1.0f);
        VF mx = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(lfin, 16)), two255, m1);   /* R: +u */
        VF my = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_and_si256(hfin, _mm256_set1_epi32(0xFFFF))), two255, m1);   /* G: image up */
        VF mz = _mm256_sqrt_ps(_mm256_max_ps(_mm256_fnmadd_ps(my, my, _mm256_fnmadd_ps(mx, mx, one)), zero));   /* z reconstructed */
        VF qx = _mm256_fmadd_ps(mx, tx, _mm256_fmadd_ps(my, bx, _mm256_mul_ps(mz, nx)));
        VF qy = _mm256_fmadd_ps(mx, ty, _mm256_fmadd_ps(my, by, _mm256_mul_ps(mz, ny)));
        VF qz = _mm256_fmadd_ps(mx, tz, _mm256_fmadd_ps(my, bz, _mm256_mul_ps(mz, nz)));
        VF ql = v_rsqrt(_mm256_max_ps(_mm256_fmadd_ps(qx, qx, _mm256_fmadd_ps(qy, qy, _mm256_mul_ps(qz, qz))), eps));
        nx = _mm256_mul_ps(qx, ql); ny = _mm256_mul_ps(qy, ql); nz = _mm256_mul_ps(qz, ql);
    }
    tex_fetch(mat->tex->texels, &tc, mask, &lfin, &hfin);
    const VF inv255 = _mm256_set1_ps(1.0f / 255.0f);
    if (alphaTest) {                                         /* R5: the bilinear alpha, linear */
        VF a = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(hfin, 16)), inv255);
        mask = _mm256_and_si256(mask, _mm256_castps_si256(_mm256_cmp_ps(a, _mm256_set1_ps(mat->alphaRef), _CMP_GE_OQ)));
        mb = mask_bitsi(mask);
        if (!mb) return 0;
    }
    {
        const VI MLO = _mm256_set1_epi32(0xFFFF);
        VF ar = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(lfin, 16)), inv255);
        VF ag = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(hfin, MLO)), inv255);
        VF ab = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(lfin, MLO)), inv255);
        ar = _mm256_mul_ps(ar, ar); ag = _mm256_mul_ps(ag, ag); ab = _mm256_mul_ps(ab, ab);   /* gamma 2 -> linear */

        /* ---- the sun (R9, R10). The terminator fade is exactly 0 at
           N.L <= grazeLo, and both sun terms scale by it, so a block with no
           sun-facing lane skips the world position, the shadow lookup and
           the specular. ---- */
        const VF Lx = _mm256_set1_ps(g_sunDir.x), Ly = _mm256_set1_ps(g_sunDir.y), Lz = _mm256_set1_ps(g_sunDir.z);
        VF ndl = _mm256_max_ps(_mm256_fmadd_ps(nx, Lx, _mm256_fmadd_ps(ny, Ly, _mm256_mul_ps(nz, Lz))), zero);
        const VF sunv = _mm256_and_ps(_mm256_castsi256_ps(mask), _mm256_cmp_ps(ndl, _mm256_set1_ps(LIT_grazeLo), _CMP_GT_OQ));
        VF sh = zero, spec = zero;
        if (mask_bits(sunv)) {
            VF px = _mm256_mul_ps(SH_PLANE(t, pgx[0], pgy[0], pc[0]), w);
            VF py = _mm256_mul_ps(SH_PLANE(t, pgx[1], pgy[1], pc[1]), w);
            VF pz = _mm256_mul_ps(SH_PLANE(t, pgx[2], pgy[2], pc[2]), w);
            VF sinT = _mm256_sqrt_ps(_mm256_max_ps(_mm256_fnmadd_ps(ndl, ndl, one), zero));
            VF tanT = _mm256_min_ps(_mm256_mul_ps(sinT, v_rcp(_mm256_max_ps(ndl, _mm256_set1_ps(0.08f)))), _mm256_set1_ps(6.0f));
            sh = shadow_sample(px, py, pz, nx, ny, nz, tanT, sunv);
            VF gt = _mm256_mul_ps(_mm256_sub_ps(ndl, _mm256_set1_ps(LIT_grazeLo)), _mm256_set1_ps(1.0f / (LIT_grazeHi - LIT_grazeLo)));
            gt = _mm256_min_ps(_mm256_max_ps(gt, zero), one);
            gt = _mm256_mul_ps(_mm256_mul_ps(gt, gt), _mm256_fnmadd_ps(_mm256_set1_ps(2.0f), gt, _mm256_set1_ps(3.0f)));   /* smoothstep */
            sh = _mm256_mul_ps(sh, gt);
            if (mask_bits(_mm256_and_ps(sunv, _mm256_cmp_ps(sh, zero, _CMP_NEQ_OQ)))) {   /* a lit lane: Blinn-Phong */
                VF vx = _mm256_sub_ps(_mm256_set1_ps(g_camPos.x), px);
                VF vy = _mm256_sub_ps(_mm256_set1_ps(g_camPos.y), py);
                VF vz = _mm256_sub_ps(_mm256_set1_ps(g_camPos.z), pz);
                VF vl = v_rsqrt(_mm256_max_ps(_mm256_fmadd_ps(vx, vx, _mm256_fmadd_ps(vy, vy, _mm256_mul_ps(vz, vz))), eps));
                VF hx = _mm256_fmadd_ps(vx, vl, Lx), hy = _mm256_fmadd_ps(vy, vl, Ly), hz = _mm256_fmadd_ps(vz, vl, Lz);   /* V + L */
                VF hl = v_rsqrt(_mm256_max_ps(_mm256_fmadd_ps(hx, hx, _mm256_fmadd_ps(hy, hy, _mm256_mul_ps(hz, hz))), eps));
                VF ndh = _mm256_max_ps(_mm256_mul_ps(_mm256_fmadd_ps(nx, hx, _mm256_fmadd_ps(ny, hy, _mm256_mul_ps(nz, hz))), hl), zero);
                spec = _mm256_mul_ps(v_pow(ndh, _mm256_set1_ps(mat->shininess)), _mm256_set1_ps(mat->ks));
                spec = _mm256_mul_ps(spec, _mm256_min_ps(_mm256_mul_ps(ndl, _mm256_set1_ps(8.0f)), one));
            }
        }
        /* ---- hemisphere ambient: sky colour up, ground colour down ---- */
        const VF hemi = _mm256_fmadd_ps(ny, _mm256_set1_ps(0.5f), _mm256_set1_ps(0.5f));
        const VF lit = _mm256_mul_ps(ndl, sh), ssh = _mm256_mul_ps(spec, sh);
        VF out[3];
        const VF alb[3] = { ar, ag, ab };
        for (int c = 0; c < 3; c++) {
            const float g0 = LIT_ambGround[c] * LIT_ambScale, s0 = LIT_ambSky[c] * LIT_ambScale;
            VF amb = _mm256_fmadd_ps(_mm256_set1_ps(s0 - g0), hemi, _mm256_set1_ps(g0));
            VF sun = _mm256_set1_ps(LIT_sunColor[c]);
            /* albedo * (sun * N.L * shadow + ambient) + sun * specular * shadow; the specular is not tinted */
            out[c] = _mm256_fmadd_ps(alb[c], _mm256_fmadd_ps(sun, lit, amb), _mm256_mul_ps(sun, ssh));
        }
        span_store(dst, mask, pack_color(out[0], out[1], out[2]));
    }
    return mb;
}

/* ============================================================================
   Sky (R13): gradient, sun disc + aureole + glow, the cloud plane
   ============================================================================ */
FORCE_INLINE VF cloud_fetch(VF u, VF v, VI mask)
{
    const VF res = _mm256_set1_ps((float)CLOUD_RES);
    VF fu = _mm256_mul_ps(u, res), fv = _mm256_mul_ps(v, res);
    VF flu = _mm256_floor_ps(fu), flv = _mm256_floor_ps(fv);
    VF fx = _mm256_sub_ps(fu, flu), fy = _mm256_sub_ps(fv, flv);
    const VI wrap = _mm256_set1_epi32(CLOUD_RES - 1), onei = _mm256_set1_epi32(1), zero = _mm256_setzero_si256(), mFF = _mm256_set1_epi32(255);
    VI x0 = _mm256_and_si256(_mm256_cvtps_epi32(flu), wrap), y0 = _mm256_and_si256(_mm256_cvtps_epi32(flv), wrap);
    VI x1 = _mm256_and_si256(_mm256_add_epi32(x0, onei), wrap), y1 = _mm256_and_si256(_mm256_add_epi32(y0, onei), wrap);
    VI r0 = _mm256_slli_epi32(y0, 9), r1 = _mm256_slli_epi32(y1, 9);   /* * CLOUD_RES */
    const int* base = (const int*)g_cloudDensity;           /* byte scale: each lane reads a dword, keeps its low byte */
    VF f00 = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_mask_i32gather_epi32(zero, base, _mm256_add_epi32(r0, x0), mask, 1), mFF));
    VF f01 = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_mask_i32gather_epi32(zero, base, _mm256_add_epi32(r0, x1), mask, 1), mFF));
    VF f10 = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_mask_i32gather_epi32(zero, base, _mm256_add_epi32(r1, x0), mask, 1), mFF));
    VF f11 = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_mask_i32gather_epi32(zero, base, _mm256_add_epi32(r1, x1), mask, 1), mFF));
    VF top = _mm256_fmadd_ps(_mm256_sub_ps(f01, f00), fx, f00);
    VF bot = _mm256_fmadd_ps(_mm256_sub_ps(f11, f10), fx, f10);
    return _mm256_mul_ps(_mm256_fmadd_ps(_mm256_sub_ps(bot, top), fy, top), _mm256_set1_ps(1.0f / 255.0f));
}

FORCE_INLINE void shade_sky8(VI mask, VF X, VF Y, uint32_t* dst)
{
    const VF one = _mm256_set1_ps(1.0f), zero = _mm256_setzero_ps();
    VF dx = _mm256_fmadd_ps(_mm256_set1_ps(g_rayB.x), X, _mm256_fmadd_ps(_mm256_set1_ps(g_rayC.x), Y, _mm256_set1_ps(g_rayA.x)));
    VF dy = _mm256_fmadd_ps(_mm256_set1_ps(g_rayB.y), X, _mm256_fmadd_ps(_mm256_set1_ps(g_rayC.y), Y, _mm256_set1_ps(g_rayA.y)));
    VF dz = _mm256_fmadd_ps(_mm256_set1_ps(g_rayB.z), X, _mm256_fmadd_ps(_mm256_set1_ps(g_rayC.z), Y, _mm256_set1_ps(g_rayA.z)));
    VF il = v_rsqrt(_mm256_fmadd_ps(dx, dx, _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dz, dz))));
    dx = _mm256_mul_ps(dx, il); dy = _mm256_mul_ps(dy, il); dz = _mm256_mul_ps(dz, il);

    VF tg = _mm256_sqrt_ps(_mm256_max_ps(dy, zero));         /* most of the gradient near the horizon */
    VF down = _mm256_min_ps(_mm256_mul_ps(_mm256_max_ps(_mm256_sub_ps(zero, dy), zero), _mm256_set1_ps(5.0f)), one);
    VF sd = _mm256_max_ps(_mm256_fmadd_ps(dx, _mm256_set1_ps(g_sunDir.x), _mm256_fmadd_ps(dy, _mm256_set1_ps(g_sunDir.y),
                                         _mm256_mul_ps(dz, _mm256_set1_ps(g_sunDir.z)))), zero);
    VF disc = _mm256_mul_ps(_mm256_sub_ps(sd, _mm256_set1_ps(g_sunCosOuter)), _mm256_set1_ps(g_sunEdgeInv));
    disc = _mm256_min_ps(_mm256_max_ps(disc, zero), one);
    disc = _mm256_mul_ps(_mm256_mul_ps(disc, disc), _mm256_fnmadd_ps(_mm256_set1_ps(2.0f), disc, _mm256_set1_ps(3.0f)));
    VF sun = _mm256_add_ps(_mm256_mul_ps(v_pow(sd, _mm256_set1_ps(SKY_haloPow)), _mm256_set1_ps(SKY_haloGain)),
                           _mm256_mul_ps(v_pow(sd, _mm256_set1_ps(SKY_glowPow)), _mm256_set1_ps(SKY_glowGain)));
    static const float tint[3] = { 1.00f, 0.92f, 0.78f };
    VF c[3];
    for (int k = 0; k < 3; k++) {
        c[k] = _mm256_fmadd_ps(_mm256_set1_ps(SKY_zenith[k] - SKY_horizon[k]), tg, _mm256_set1_ps(SKY_horizon[k]));
        c[k] = _mm256_fmadd_ps(_mm256_sub_ps(_mm256_set1_ps(SKY_ground[k]), c[k]), down, c[k]);
        c[k] = _mm256_fmadd_ps(sun, _mm256_set1_ps(tint[k]), c[k]);                          /* aureole + glow, additive */
        c[k] = _mm256_fmadd_ps(_mm256_sub_ps(_mm256_set1_ps(SKY_sunColor[k]), c[k]), disc, c[k]);   /* the disc occludes */
    }
    VI cm = _mm256_and_si256(mask, _mm256_castps_si256(_mm256_cmp_ps(dy, _mm256_set1_ps(0.01f), _CMP_GT_OQ)));
    if (mask_bitsi(cm)) {                                    /* rays going up hit the cloud plane */
        VF tt = _mm256_mul_ps(_mm256_set1_ps(g_cloudPlaneY - g_camPos.y), v_rcp(_mm256_max_ps(dy, _mm256_set1_ps(0.01f))));
        VF cu = _mm256_mul_ps(_mm256_fmadd_ps(dx, tt, _mm256_set1_ps(g_camPos.x)), _mm256_set1_ps(SKY_cloudScale));
        VF cv = _mm256_mul_ps(_mm256_fmadd_ps(dz, tt, _mm256_set1_ps(g_camPos.z)), _mm256_set1_ps(SKY_cloudScale));
        cu = _mm256_sub_ps(cu, _mm256_floor_ps(cu));
        cv = _mm256_sub_ps(cv, _mm256_floor_ps(cv));
        VF dens = cloud_fetch(cu, cv, cm);
        VF fade = v_rcp(_mm256_fmadd_ps(_mm256_mul_ps(tt, tt), _mm256_set1_ps(2e-5f), one));
        VF a = _mm256_and_ps(_mm256_mul_ps(dens, fade), _mm256_castsi256_ps(cm));
        VF cb = _mm256_fnmadd_ps(dens, _mm256_set1_ps(0.55f), _mm256_set1_ps(1.35f));   /* darker core */
        static const float ctint[3] = { 1.00f, 0.99f, 0.97f };
        for (int k = 0; k < 3; k++)
            c[k] = _mm256_fmadd_ps(_mm256_sub_ps(_mm256_mul_ps(cb, _mm256_set1_ps(ctint[k])), c[k]), a, c[k]);
    }
    span_store(dst, mask, pack_color(c[0], c[1], c[2]));
}

/* ============================================================================
   raster_tri: one triangle into one tile. mode 0 = depth prepass (opaque
   only), mode 1 = shade (LE for opaques, LT + depth write for cut-outs).
   ============================================================================ */
FORCE_INLINE void raster_tri(const Tri* t, int tx0, int ty0, float* tz, uint32_t* cbuf, int mode)
{
    const int masked = t->masked;
    if (mode == 0 && masked) return;                         /* a cut-out cannot resolve depth without its texture */
    const int minx = t->minx > tx0 ? t->minx : tx0, maxx = t->maxx < tx0 + TILE_SIZE - 1 ? t->maxx : tx0 + TILE_SIZE - 1;
    const int miny = t->miny > ty0 ? t->miny : ty0, maxy = t->maxy < ty0 + TILE_SIZE - 1 ? t->maxy : ty0 + TILE_SIZE - 1;
    if (minx > maxx || miny > maxy) return;
    const int bx0 = (minx - tx0) / SPAN_W, bx1 = (maxx - tx0) / SPAN_W;
    const int by0 = (miny - ty0) / SPAN_H, by1 = (maxy - ty0) / SPAN_H;

    /* Hierarchical classification: for big walks, settle whole 16x16-pixel
       cells (4 blocks across, 8 block rows down) from each edge's extreme
       corner values. A cell whose best corner fails an edge is EMPTY and its
       blocks are skipped; a cell whose worst corner passes every edge is FULL
       and its blocks skip the edge test. Exact: the edges are integer and
       linear, so their extremes over a cell sit at its corners. */
    unsigned cellFull = 0, cellEmpty = 0;
    unsigned char rowEmpty[4] = { 0, 0, 0, 0 };
    int hier = 0;
    if ((bx1 - bx0 + 1) * (by1 - by0 + 1) >= HIER_MIN_BLOCKS) {
        const int cx0 = bx0 >> 2, cx1 = bx1 >> 2, cy0 = by0 >> 3, cy1 = by1 >> 3;
        int64_t aMin[3][4], aMax[3][4], bMin[3][4], bMax[3][4];
        for (int k = 0; k < 3; k++) {
            const int64_t A = t->A[k], B = t->B[k];
            const int64_t a15 = A * 15, b15 = B * 15;
            int64_t va = A * (int64_t)(tx0 + cx0 * 16 - t->minx);   /* first column of the first cell */
            for (int cx = cx0; cx <= cx1; cx++, va += A * 16) { aMin[k][cx] = va + (a15 < 0 ? a15 : 0); aMax[k][cx] = va + (a15 < 0 ? 0 : a15); }
            int64_t vb = B * (int64_t)(ty0 + cy0 * 16 - t->miny) + t->C[k];
            for (int cy = cy0; cy <= cy1; cy++, vb += B * 16) { bMin[k][cy] = vb + (b15 < 0 ? b15 : 0); bMax[k][cy] = vb + (b15 < 0 ? 0 : b15); }
        }
        for (int cy = cy0; cy <= cy1; cy++) {
            unsigned rowAll = 1;
            for (int cx = cx0; cx <= cx1; cx++) {
                const unsigned bit = 1u << (cy * 4 + cx);
                int full = 1, empty = 0;
                for (int k = 0; k < 3; k++) {
                    if (aMax[k][cx] + bMax[k][cy] < 0) { empty = 1; break; }
                    if (aMin[k][cx] + bMin[k][cy] < 0) full = 0;
                }
                if (empty) cellEmpty |= bit;
                else { rowAll = 0; if (full) cellFull |= bit; }
            }
            rowEmpty[cy] = (unsigned char)rowAll;
        }
        hier = 1;
    }

    const VF lx = _mm256_load_ps(SPAN_LX), ly = _mm256_load_ps(SPAN_LY);
    const VI lxi = _mm256_load_si256((const __m256i*)SPAN_LXI), lyi = _mm256_load_si256((const __m256i*)SPAN_LYI);
    /* per-lane edge deltas A*lx + B*ly, once per triangle */
    const VI dl0 = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_set1_epi32(t->A[0]), lxi), _mm256_mullo_epi32(_mm256_set1_epi32(t->B[0]), lyi));
    const VI dl1 = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_set1_epi32(t->A[1]), lxi), _mm256_mullo_epi32(_mm256_set1_epi32(t->B[1]), lyi));
    const VI dl2 = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_set1_epi32(t->A[2]), lxi), _mm256_mullo_epi32(_mm256_set1_epi32(t->B[2]), lyi));
    const VI minus1 = _mm256_set1_epi32(-1);
    const VF vZx = _mm256_set1_ps(t->zgx), vZy = _mm256_set1_ps(t->zgy), vZc = _mm256_set1_ps(t->zc);
    const VF vZmin = _mm256_set1_ps(t->minz);               /* depth floor (R3) */
    const unsigned xmFirst = SPAN_XREP(SPAN_RUN(minx - (tx0 + bx0 * SPAN_W), SPAN_W - 1));
    const unsigned xmLast = SPAN_XREP(SPAN_RUN(0, maxx - (tx0 + bx1 * SPAN_W)));
    /* opaques in the shade pass find their own prepass depth: less-or-equal;
       the prepass and cut-outs are a genuine less-than */
    const int lessEq = mode == 1 && !masked;

    for (int by = by0; by <= by1; by++) {
        if (hier && rowEmpty[by >> 3]) { by |= 7; continue; }   /* the whole cell row is outside */
        const int py0 = ty0 + by * SPAN_H;
        const int loy = miny > py0 ? miny - py0 : 0, hiy = maxy < py0 + SPAN_H - 1 ? maxy - py0 : SPAN_H - 1;
        const unsigned ym = SPAN_YEXP[SPAN_RUN(loy, hiy)];
        const size_t rowOff = SPAN_OFF(0, by);
        const size_t rowCol = (size_t)(py0 - ty0) * TILE_SIZE;
        const VF Y = _mm256_add_ps(_mm256_set1_ps((float)(py0 - t->miny) + 0.5f), ly);
        const VF zy = _mm256_mul_ps(vZy, Y);
        const int64_t yoff = py0 - t->miny;
        const int64_t rowC0 = t->C[0] + (int64_t)t->B[0] * yoff, rowC1 = t->C[1] + (int64_t)t->B[1] * yoff, rowC2 = t->C[2] + (int64_t)t->B[2] * yoff;
        for (int bx = bx0; bx <= bx1; bx++) {
            int hAcc = 0;
            if (hier) {
                const unsigned ci = (unsigned)(((by >> 3) << 2) | (bx >> 2));
                if ((cellEmpty >> ci) & 1) { bx |= 3; continue; }   /* cell outside: to its last block */
                hAcc = (int)((cellFull >> ci) & 1);                  /* cell inside: no edge test */
            }
            const int px0 = tx0 + bx * SPAN_W;
            unsigned bbm = ym;
            if (bx == bx0) bbm &= xmFirst;
            if (bx == bx1) bbm &= xmLast;
            unsigned mk;
            if (hAcc) mk = bbm;
            else {
                const int64_t xoff = px0 - t->minx;
                int64_t v0 = rowC0 + (int64_t)t->A[0] * xoff, v1 = rowC1 + (int64_t)t->A[1] * xoff, v2 = rowC2 + (int64_t)t->A[2] * xoff;
                v0 = v0 > EDGE_SAT ? EDGE_SAT : (v0 < -EDGE_SAT ? -EDGE_SAT : v0);   /* sign-preserving saturation */
                v1 = v1 > EDGE_SAT ? EDGE_SAT : (v1 < -EDGE_SAT ? -EDGE_SAT : v1);
                v2 = v2 > EDGE_SAT ? EDGE_SAT : (v2 < -EDGE_SAT ? -EDGE_SAT : v2);
                VI in = _mm256_cmpgt_epi32(_mm256_add_epi32(_mm256_set1_epi32((int32_t)v0), dl0), minus1);   /* e >= 0 */
                in = _mm256_and_si256(in, _mm256_cmpgt_epi32(_mm256_add_epi32(_mm256_set1_epi32((int32_t)v1), dl1), minus1));
                in = _mm256_and_si256(in, _mm256_cmpgt_epi32(_mm256_add_epi32(_mm256_set1_epi32((int32_t)v2), dl2), minus1));
                mk = (unsigned)mask_bitsi(in) & bbm;
            }
            if (!mk) continue;
            const VF X = _mm256_add_ps(_mm256_set1_ps((float)(px0 - t->minx) + 0.5f), lx);
            const VF zs = _mm256_max_ps(_mm256_add_ps(_mm256_fmadd_ps(vZx, X, vZc), zy), vZmin);
            const float* zp = tz + rowOff + (size_t)bx * 8;
            const VF zold = _mm256_load_ps(zp);
            const VF pass = lessEq ? _mm256_cmp_ps(zs, zold, _CMP_LE_OQ) : _mm256_cmp_ps(zs, zold, _CMP_LT_OQ);
            const VF cov = _mm256_and_ps(pass, _mm256_castsi256_ps(mask_vec(mk)));
            const int cb = mask_bits(cov);
            if (!cb) continue;
            if (mode == 0)
                _mm256_store_ps((float*)zp, _mm256_blendv_ps(zold, zs, cov));   /* per-pixel MIN: order-independent */
            else {
                const int sm = shade8(t, _mm256_castps_si256(cov), cb, X, Y, cbuf + rowCol + (size_t)(bx * SPAN_W), masked);
                if (masked && sm)                                                /* cut-outs own their depth */
                    _mm256_store_ps((float*)zp, _mm256_blendv_ps(zold, zs, _mm256_castsi256_ps(mask_vec((unsigned)sm))));
            }
        }
    }
}

/* ============================================================================
   job_tile
   ============================================================================ */
static void job_tile(int tile)
{
    const Uint64 cyc0 = SDL_GetPerformanceCounter();
    const int tx0 = (tile % g_tilesX) << TILE_SHIFT, ty0 = (tile / g_tilesX) << TILE_SHIFT;
    SDL_ALIGNED(32) float tz[TILE_SIZE * TILE_SIZE];       /* tile-local depth, swizzled 4x2 blocks */
    SDL_ALIGNED(32) uint32_t tcol[TILE_SIZE * TILE_SIZE];  /* tile-local colour, row-major */

    /* the chunks that binned into this tile, ascending, from its chunk bitmap */
    int* const chs = SDL_stack_alloc(int, g_ntriChunks);
    int nch = 0;
    SDL_AtomicInt* const words = g_tileChunkBits + (size_t)tile * g_chunkWords;
    for (int wi = 0; wi < g_chunkWords; wi++)
        for (uint32_t bits = (uint32_t)SDL_GetAtomicInt(&words[wi]); bits; bits &= bits - 1)
            chs[nch++] = wi * 32 + (int)ctz32(bits);

    const VF onez = _mm256_set1_ps(1.0f);
    for (int i = 0; i < TILE_SIZE * TILE_SIZE; i += 8) _mm256_store_ps(tz + i, onez);

    /* depth prepass, then the shade pass -- both in chunk order */
    for (int pass = 0; pass < 2; pass++)
        for (int li = 0; li < nch; li++) {
            const Bin* b = &g_bins[(size_t)chs[li] * g_ntiles + tile];
            const int n = b->stamp == g_frameStamp ? b->count : 0;
            for (int i = 0; i < n; i++) raster_tri(&g_tris[b->items[i]], tx0, ty0, tz, tcol, pass);
        }

    /* the sky, wherever the depth is still exactly the clear value */
    const VF lx = _mm256_load_ps(SPAN_LX), ly = _mm256_load_ps(SPAN_LY);
    for (int by = 0; by < SPAN_BY; by++) {
        const int py0 = ty0 + by * SPAN_H;
        if (py0 >= g_height) break;
        const int rows = g_height - py0;
        const unsigned ym = SPAN_YEXP[(1u << (rows >= SPAN_H ? SPAN_H : rows)) - 1u];
        for (int bx = 0; bx < SPAN_BX; bx++) {
            const int px0 = tx0 + bx * SPAN_W;
            if (px0 >= g_width) break;
            const int cols = g_width - px0;
            const unsigned wm = SPAN_XREP((1u << (cols >= SPAN_W ? SPAN_W : cols)) - 1u) & ym;
            const VF sky = _mm256_and_ps(_mm256_cmp_ps(_mm256_load_ps(tz + SPAN_OFF(bx, by)), onez, _CMP_EQ_OQ),
                                         _mm256_castsi256_ps(mask_vec(wm)));
            if (!mask_bits(sky)) continue;
            const VF X = _mm256_add_ps(_mm256_set1_ps((float)px0 + 0.5f), lx);
            const VF Y = _mm256_add_ps(_mm256_set1_ps((float)py0 + 0.5f), ly);
            shade_sky8(_mm256_castps_si256(sky), X, Y, tcol + (size_t)(py0 - ty0) * TILE_SIZE + (size_t)(px0 - tx0));
        }
    }

    /* Stream the tile out in whole 256-byte rows. The frame's row pitch is a
       whole number of tiles, so each row starts on a 32-byte boundary; the
       columns past the window's right edge land in that pitch, unseen. */
    const int ry1 = ty0 + TILE_SIZE < g_height ? ty0 + TILE_SIZE : g_height;
    for (int y = ty0; y < ry1; y++) {
        uint32_t* dst = g_frame + (size_t)y * g_frameStride + tx0;
        const uint32_t* src = tcol + (size_t)(y - ty0) * TILE_SIZE;
        for (int x = 0; x < TILE_SIZE; x += 8)
            _mm256_stream_si256((__m256i*)(dst + x), _mm256_load_si256((const __m256i*)(src + x)));
    }
    _mm_sfence();
    g_tileCost[tile] = (unsigned)(SDL_GetPerformanceCounter() - cyc0);   /* next frame's costly-first order */
    SDL_stack_free(chs);
}
