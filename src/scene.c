/* scene.c -- loading (R2): image decoding (stb_image), mip chains, a minimal
   glTF 2.0 reader for Sponza, normal maps, and the cloud texture.

   Load rules that change the image (docs/RENDERING_SPEC.md, R2):
   - the last two indices of every triangle are swapped once (glTF fronts are
     CCW, this renderer's visible side is cross(e2, e1));
   - the node's uniform scale and translation are baked into the positions;
   - baseColorFactor is baked into the texels as sqrt(factor);
   - doubleSided decides back-face culling per triangle;
   - MASK materials are alpha-tested and cast no shadow;
   - the metallic-roughness maps are averaged into a Blinn-Phong pair. */
#include "core.h"

/* stb_image's interface (the decoder itself is compiled in image.c) */
#define STBI_NO_STDIO
#include "stb_image.h"

static void fatal(const char* msg)
{
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Sponza (simple)", msg, NULL);
    exit(1);
}

/* decode a JPEG/PNG under the asset directory into a malloc'd ARGB array
   (0xAARRGGBB, the byte order the shader's texel fetch expects) */
static uint32_t* image_decode(const char* relPath, unsigned* outW, unsigned* outH)
{
    char path[PATH_LEN + 256];
    snprintf(path, sizeof path, "%s%s", g_assetDir, relPath);
    size_t len = 0;
    unsigned char* file = (unsigned char*)SDL_LoadFile(path, &len);
    if (!file) return NULL;
    int w, h, n;
    unsigned char* rgba = stbi_load_from_memory(file, (int)len, &w, &h, &n, 4);
    SDL_free(file);
    if (!rgba) return NULL;
    uint32_t* px = (uint32_t*)malloc((size_t)w * h * 4);
    for (size_t i = 0, cnt = (size_t)w * h; px && i < cnt; i++) {
        const unsigned char* p = rgba + i * 4;
        px[i] = ((uint32_t)p[3] << 24) | ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | p[2];
    }
    stbi_image_free(rgba);
    *outW = (unsigned)w; *outH = (unsigned)h;
    return px;
}

/* ============================================================================
   Textures: the whole mip chain in one allocation, built by a rounded 2x2 box
   in the stored (sqrt-encoded) space, alpha carried along (R4).
   ============================================================================ */
static Texture* texture_from_pixels(const uint32_t* level0, int w, int h)
{
    Texture* t = (Texture*)calloc(1, sizeof(Texture));
    int nm = 0; size_t total = 0;
    for (int cw = w, ch = h; nm < MAX_MIPS; nm++) {
        t->mw[nm] = cw; t->mh[nm] = ch; t->moff[nm] = (int32_t)total;
        total += (size_t)cw * ch;
        if (cw == 1 && ch == 1) { nm++; break; }
        cw = cw > 1 ? cw >> 1 : 1;
        ch = ch > 1 ? ch >> 1 : 1;
    }
    t->w0 = w; t->h0 = h; t->nmips = nm;
    /* +64 B: the bilinear fetch reads texel pairs as 64-bit elements, so the
       last texel's right neighbour must stay in bounds */
    t->texels = (uint32_t*)SDL_aligned_alloc(64, total * 4 + 64);
    memset(t->texels + total, 0, 64);
    memcpy(t->texels, level0, (size_t)w * h * 4);
    for (int m = 1; m < nm; m++) {
        const uint32_t* src = t->texels + t->moff[m - 1];
        uint32_t* dst = t->texels + t->moff[m];
        int sw = t->mw[m - 1], sh = t->mh[m - 1], dw = t->mw[m], dh = t->mh[m];
        for (int y = 0; y < dh; y++) {
            int y0 = y * 2, y1 = y0 + 1 < sh ? y0 + 1 : y0;
            for (int x = 0; x < dw; x++) {
                int x0 = x * 2, x1 = x0 + 1 < sw ? x0 + 1 : x0;
                uint32_t c00 = src[y0 * sw + x0], c01 = src[y0 * sw + x1];
                uint32_t c10 = src[y1 * sw + x0], c11 = src[y1 * sw + x1];
                uint32_t o = 0;
                for (int s = 0; s < 32; s += 8)   /* per channel: sum of four, +2, >>2 */
                    o |= ((((c00 >> s) & 255) + ((c01 >> s) & 255) + ((c10 >> s) & 255) + ((c11 >> s) & 255) + 2) >> 2) << s;
                dst[y * dw + x] = o;
            }
        }
    }
    for (int m = nm; m < MAX_MIPS; m++) {        /* pad the tables with the last level */
        t->mw[m] = t->mw[nm - 1]; t->mh[m] = t->mh[nm - 1]; t->moff[m] = t->moff[nm - 1];
    }
    return t;
}

/* base colour: baseColorFactor baked in as sqrt(factor), since texels are sqrt-encoded */
static Texture* texture_load_image(const char* relPath, float mr, float mg, float mb)
{
    unsigned w, h;
    uint32_t* px = image_decode(relPath, &w, &h);
    if (!px) return NULL;
    if (mr != 1.0f || mg != 1.0f || mb != 1.0f) {
        float sr = sqrtf(clampf(mr, 0, 1)), sg = sqrtf(clampf(mg, 0, 1)), sb = sqrtf(clampf(mb, 0, 1));
        for (size_t i = 0, n = (size_t)w * h; i < n; i++) {
            uint32_t c = px[i];
            uint32_t r = (uint32_t)(((c >> 16) & 255) * sr + 0.5f);
            uint32_t g = (uint32_t)(((c >> 8) & 255) * sg + 0.5f);
            uint32_t b = (uint32_t)((c & 255) * sb + 0.5f);
            px[i] = (c & 0xFF000000u) | (r << 16) | (g << 8) | b;
        }
    }
    Texture* t = texture_from_pixels(px, (int)w, (int)h);
    free(px);
    return t;
}

static Texture* texture_solid(float r, float g, float b)
{
    uint32_t ri = (uint32_t)(sqrtf(clampf(r, 0, 1)) * 255.0f + 0.5f);
    uint32_t gi = (uint32_t)(sqrtf(clampf(g, 0, 1)) * 255.0f + 0.5f);
    uint32_t bi = (uint32_t)(sqrtf(clampf(b, 0, 1)) * 255.0f + 0.5f);
    uint32_t pix[16];
    for (int i = 0; i < 16; i++) pix[i] = 0xFF000000u | (ri << 16) | (gi << 8) | bi;
    return texture_from_pixels(pix, 4, 4);
}

static char* read_entire_file(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)len + 2);
    const size_t got = buf && len >= 0 ? fread(buf, 1, (size_t)len, f) : 0;
    fclose(f);
    if (!buf || got != (size_t)len) { free(buf); return NULL; }
    buf[len] = '\n'; buf[len + 1] = 0;           /* the tokenizer may run off the last token */
    return buf;
}

/* ============================================================================
   JSON tokenizer: a first-child / next-sibling token tree over the document.
   Objects hold key tokens ('K') whose child is the value.
   ============================================================================ */
typedef struct JTok { char kind; int start, end, kid, sib; } JTok;   /* kind: { [ " K # */
typedef struct Gltf {
    char* json; JTok* tok; int ntok, captok;
    uint8_t* bin;
    int accessors, bufferViews, meshes, materials, textures, images, nodes, scenes;
} Gltf;

static int jtok_new(Gltf* g, char kind, int start)
{
    if (g->ntok == g->captok) {
        g->captok *= 2;
        g->tok = (JTok*)realloc(g->tok, (size_t)g->captok * sizeof(JTok));
    }
    JTok* t = &g->tok[g->ntok];
    t->kind = kind; t->start = start; t->end = -1; t->kid = -1; t->sib = -1;
    return g->ntok++;
}
static char* jws(char* p) { while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++; return p; }
static char* jparse_string(Gltf* g, char* p, int* out)
{
    int tk = jtok_new(g, '"', (int)(p + 1 - g->json));
    p++;
    while (*p && *p != '"') { if (*p == '\\' && p[1]) p++; p++; }
    g->tok[tk].end = (int)(p - g->json);
    *out = tk;
    return *p ? p + 1 : p;
}
/* g->tok may be reallocated by children: always re-index, never hold a JTok* */
static char* jparse_value(Gltf* g, char* p, int* out)
{
    p = jws(p);
    if (*p == '{' || *p == '[') {
        const char open = *p, close = open == '{' ? '}' : ']';
        int tk = jtok_new(g, open, (int)(p - g->json)), last = -1;
        p = jws(p + 1);
        while (*p && *p != close) {
            int item;
            if (open == '{') {
                int val;
                p = jparse_string(g, p, &item);
                g->tok[item].kind = 'K';
                p = jws(p);
                if (*p == ':') p++;
                p = jparse_value(g, p, &val);
                g->tok[item].kid = val;
            } else p = jparse_value(g, p, &item);
            if (last < 0) g->tok[tk].kid = item; else g->tok[last].sib = item;
            last = item;
            p = jws(p);
            if (*p == ',') p = jws(p + 1);
        }
        g->tok[tk].end = (int)(p - g->json);
        *out = tk;
        return *p ? p + 1 : p;
    }
    if (*p == '"') return jparse_string(g, p, out);
    int tk = jtok_new(g, '#', (int)(p - g->json));       /* number / true / false / null */
    while (*p && *p != ',' && *p != '}' && *p != ']' && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') p++;
    g->tok[tk].end = (int)(p - g->json);
    *out = tk;
    return p;
}
static int jget(const Gltf* g, int obj, const char* key)
{
    if (obj < 0 || g->tok[obj].kind != '{') return -1;
    size_t kl = strlen(key);
    for (int k = g->tok[obj].kid; k >= 0; k = g->tok[k].sib)
        if ((size_t)(g->tok[k].end - g->tok[k].start) == kl && memcmp(g->json + g->tok[k].start, key, kl) == 0)
            return g->tok[k].kid;
    return -1;
}
static int jat(const Gltf* g, int arr, int idx)
{
    if (arr < 0 || g->tok[arr].kind != '[') return -1;
    int k = g->tok[arr].kid;
    while (k >= 0 && idx-- > 0) k = g->tok[k].sib;
    return k;
}
static int jcount(const Gltf* g, int arr)
{
    int n = 0;
    if (arr >= 0 && g->tok[arr].kind == '[')
        for (int k = g->tok[arr].kid; k >= 0; k = g->tok[k].sib) n++;
    return n;
}
static double jnum(const Gltf* g, int tok, double def)
{
    return (tok < 0 || g->tok[tok].kind != '#') ? def : strtod(g->json + g->tok[tok].start, NULL);
}
static int jint(const Gltf* g, int tok, int def) { return tok < 0 ? def : (int)jnum(g, tok, def); }
static int jbool(const Gltf* g, int tok, int def)
{
    return (tok < 0 || g->tok[tok].kind != '#') ? def : g->json[g->tok[tok].start] == 't';
}
static int jstreq(const Gltf* g, int tok, const char* s)
{
    if (tok < 0 || g->tok[tok].kind != '"') return 0;
    size_t n = strlen(s);
    return (size_t)(g->tok[tok].end - g->tok[tok].start) == n && memcmp(g->json + g->tok[tok].start, s, n) == 0;
}
static void jstrz(const Gltf* g, int tok, char* out, int cap)
{
    out[0] = 0;
    if (tok < 0 || g->tok[tok].kind != '"') return;
    int n = g->tok[tok].end - g->tok[tok].start;
    if (n > cap - 1) n = cap - 1;
    memcpy(out, g->json + g->tok[tok].start, (size_t)n);
    out[n] = 0;
}

/* a strided window into the .bin buffer */
typedef struct { const uint8_t* p; int count, stride, comp; } AccView;
static AccView acc_view(const Gltf* g, int accIdx, int ncomp)
{
    AccView v; memset(&v, 0, sizeof v);
    int a = jat(g, g->accessors, accIdx);
    if (a < 0) return v;
    int bv = jat(g, g->bufferViews, jint(g, jget(g, a, "bufferView"), 0));
    v.count = jint(g, jget(g, a, "count"), 0);
    v.comp = jint(g, jget(g, a, "componentType"), 5126);
    int csize = v.comp == 5123 ? 2 : (v.comp == 5121 ? 1 : 4);
    int stride = jint(g, jget(g, bv, "byteStride"), 0);
    v.stride = stride ? stride : csize * ncomp;
    v.p = g->bin + jint(g, jget(g, bv, "byteOffset"), 0) + jint(g, jget(g, a, "byteOffset"), 0);
    return v;
}
static void gltf_image_uri(const Gltf* g, int texIdx, char* out, int cap)
{
    int tx = jat(g, g->textures, texIdx);
    int img = jat(g, g->images, jint(g, jget(g, tx, "source"), 0));
    jstrz(g, jget(g, img, "uri"), out, cap);
}

/* PBR metallic-roughness average -> Blinn-Phong (ks, shininess) (R2) */
static void spec_from_roughness(float avgRough, float avgMetal, float* outKs, float* outShininess)
{
    float r = clampf(avgRough, 0.30f, 1.0f);
    float a = r * r;
    *outShininess = clampf(2.0f / (a * a) - 2.0f, 8.0f, 256.0f);
    *outKs = clampf(LIT_specKsMin + 0.35f * (1.0f - r) * (1.0f - r) + 0.40f * avgMetal, LIT_specKsMin, 0.60f);
}

/* ============================================================================
   Normal maps (R6): decoded as linear bytes, one pool job each. A map is used
   only at its base colour's size, so one set of texel coordinates serves both.
   ============================================================================ */
typedef struct NrmReq { char uri[96]; Texture* tex; } NrmReq;
static NrmReq g_nrmReq[MAX_MATERIALS];
static int g_nrmN;
static int g_matNrm[MAX_MATERIALS];

static void job_nrm(int idx)
{
    NrmReq* r = &g_nrmReq[idx];
    char rel[160]; snprintf(rel, sizeof rel, SCENE_DIR "%s", r->uri);
    unsigned w, h;
    uint32_t* px = image_decode(rel, &w, &h);
    if (!px) return;
    r->tex = texture_from_pixels(px, (int)w, (int)h);
    free(px);
}

/* ============================================================================
   The loader
   ============================================================================ */
static void load_gltf(void)
{
    char path[PATH_LEN + 256];                  /* the asset directory plus a file name */
    Gltf G; memset(&G, 0, sizeof G);
    snprintf(path, sizeof path, "%s" SCENE_DIR "Sponza.gltf", g_assetDir);
    G.json = read_entire_file(path);
    if (!G.json) fatal(SCENE_DIR "Sponza.gltf not found next to the exe or in a parent directory.");
    G.captok = 1 << 15;
    G.tok = (JTok*)malloc((size_t)G.captok * sizeof(JTok));
    int root;
    jparse_value(&G, G.json, &root);
    G.accessors   = jget(&G, root, "accessors");
    G.bufferViews = jget(&G, root, "bufferViews");
    G.meshes      = jget(&G, root, "meshes");
    G.materials   = jget(&G, root, "materials");
    G.textures    = jget(&G, root, "textures");
    G.images      = jget(&G, root, "images");
    G.nodes       = jget(&G, root, "nodes");
    G.scenes      = jget(&G, root, "scenes");
    {
        char binName[128];
        jstrz(&G, jget(&G, jat(&G, jget(&G, root, "buffers"), 0), "uri"), binName, sizeof binName);
        snprintf(path, sizeof path, "%s" SCENE_DIR "%s", g_assetDir, binName);
        G.bin = (uint8_t*)read_entire_file(path);
        if (!G.bin) fatal("The glTF buffer (Sponza.bin) was not found.");
    }

    /* ---- materials ---- */
    uint8_t twoSided[MAX_MATERIALS] = { 0 };
    struct { char uri[96]; Texture* tex; } texCache[MAX_MATERIALS]; int ntexCache = 0;
    g_nrmN = 0;
    for (int i = 0; i < MAX_MATERIALS; i++) g_matNrm[i] = -1;
    g_nmats = jcount(&G, G.materials);
    if (g_nmats > MAX_MATERIALS) g_nmats = MAX_MATERIALS;
    if (g_nmats == 0) g_nmats = 1;
    for (int i = 0; i < g_nmats; i++) {
        Material* m = &g_mats[i];
        memset(m, 0, sizeof *m);
        int jm = jat(&G, G.materials, i);
        twoSided[i] = (uint8_t)jbool(&G, jget(&G, jm, "doubleSided"), 0);
        m->alphaMask = jstreq(&G, jget(&G, jm, "alphaMode"), "MASK");
        m->alphaRef = (float)jnum(&G, jget(&G, jm, "alphaCutoff"), 0.5);
        int pbr = jget(&G, jm, "pbrMetallicRoughness");
        float bcf[4] = { 1, 1, 1, 1 };
        int f = jget(&G, pbr, "baseColorFactor");
        if (f >= 0) for (int c = 0; c < 4; c++) bcf[c] = (float)jnum(&G, jat(&G, f, c), 1.0);

        int bct = jget(&G, pbr, "baseColorTexture");     /* deduplicated by image uri */
        if (bct >= 0) {
            char uri[96];
            gltf_image_uri(&G, jint(&G, jget(&G, bct, "index"), 0), uri, sizeof uri);
            for (int c = 0; c < ntexCache; c++)
                if (strcmp(texCache[c].uri, uri) == 0) { m->tex = texCache[c].tex; break; }
            if (!m->tex && uri[0]) {
                char rel[160]; snprintf(rel, sizeof rel, SCENE_DIR "%s", uri);
                m->tex = texture_load_image(rel, bcf[0], bcf[1], bcf[2]);
                if (m->tex && ntexCache < MAX_MATERIALS) {
                    snprintf(texCache[ntexCache].uri, sizeof texCache[0].uri, "%s", uri);
                    texCache[ntexCache++].tex = m->tex;
                }
            }
        }
        if (!m->tex) m->tex = texture_solid(bcf[0], bcf[1], bcf[2]);

        int nt = jget(&G, jm, "normalTexture");           /* collected here, decoded on the pool below */
        if (nt >= 0) {
            char uri[96];
            gltf_image_uri(&G, jint(&G, jget(&G, nt, "index"), 0), uri, sizeof uri);
            if (uri[0]) {
                int k = 0;
                while (k < g_nrmN && strcmp(g_nrmReq[k].uri, uri) != 0) k++;
                if (k == g_nrmN) { snprintf(g_nrmReq[k].uri, sizeof g_nrmReq[k].uri, "%s", uri); g_nrmReq[k].tex = NULL; g_nrmN++; }
                g_matNrm[i] = k;
            }
        }

        float rough = 0.7f, metal = 0.0f;                  /* the MR map's average: G roughness, B metallic */
        int mrt = jget(&G, pbr, "metallicRoughnessTexture");
        if (mrt >= 0) {
            char uri[96], rel[160];
            gltf_image_uri(&G, jint(&G, jget(&G, mrt, "index"), 0), uri, sizeof uri);
            snprintf(rel, sizeof rel, SCENE_DIR "%s", uri);
            unsigned w, h;
            uint32_t* px = uri[0] ? image_decode(rel, &w, &h) : NULL;
            if (px) {
                uint64_t sg = 0, sb = 0; size_t n = (size_t)w * h;
                for (size_t k = 0; k < n; k++) { sg += (px[k] >> 8) & 255; sb += px[k] & 255; }
                rough = (float)sg / (255.0f * n);
                metal = (float)sb / (255.0f * n);
                free(px);
            }
        }
        spec_from_roughness(rough, metal, &m->ks, &m->shininess);
    }
    if (g_nrmN) dispatch(PH_NRM, g_nrmN);
    for (int i = 0; i < g_nmats; i++) {
        int k = g_matNrm[i];
        if (k >= 0 && g_nrmReq[k].tex && g_nrmReq[k].tex->w0 == g_mats[i].tex->w0 && g_nrmReq[k].tex->h0 == g_mats[i].tex->h0)
            g_mats[i].nrm = g_nrmReq[k].tex;
    }
    if (jcount(&G, G.materials) == 0) {
        g_mats[0].ks = LIT_specKsMin; g_mats[0].shininess = 32.0f;
        g_mats[0].tex = texture_solid(0.7f, 0.7f, 0.7f);
        twoSided[0] = 1;
    }

    /* ---- the node: uniform scale + translation, baked into the positions ---- */
    float scl[3] = { 1, 1, 1 }, trn[3] = { 0, 0, 0 };
    int meshIdx = 0;
    {
        int scene = jat(&G, G.scenes, jint(&G, jget(&G, root, "scene"), 0));
        int node = jat(&G, G.nodes, jint(&G, jat(&G, jget(&G, scene, "nodes"), 0), 0));
        meshIdx = jint(&G, jget(&G, node, "mesh"), 0);
        int js = jget(&G, node, "scale");
        if (js >= 0) for (int c = 0; c < 3; c++) scl[c] = (float)jnum(&G, jat(&G, js, c), 1.0);
        int jt = jget(&G, node, "translation");
        if (jt >= 0) for (int c = 0; c < 3; c++) trn[c] = (float)jnum(&G, jat(&G, jt, c), 0.0);
    }

    /* ---- geometry: count, then fill ---- */
    int prims = jget(&G, jat(&G, G.meshes, meshIdx), "primitives");
    int nv = 0, nt = 0;
    for (int pr = prims >= 0 ? G.tok[prims].kid : -1; pr >= 0; pr = G.tok[pr].sib) {
        int attr = jget(&G, pr, "attributes");
        nv += jint(&G, jget(&G, jat(&G, G.accessors, jint(&G, jget(&G, attr, "POSITION"), 0)), "count"), 0);
        nt += jint(&G, jget(&G, jat(&G, G.accessors, jint(&G, jget(&G, pr, "indices"), 0)), "count"), 0) / 3;
    }
    g_verts = (Vertex*)malloc((size_t)nv * sizeof(Vertex));
    g_idx = (uint32_t*)malloc((size_t)nt * 3 * sizeof(uint32_t));
    g_triMat = (uint16_t*)malloc((size_t)nt * sizeof(uint16_t));
    g_triCullBack = (uint8_t*)calloc((size_t)nt, 1);
    g_triNoCast = (uint8_t*)calloc((size_t)nt, 1);
    g_nverts = 0; g_ntris = 0;
    for (int pr = prims >= 0 ? G.tok[prims].kid : -1; pr >= 0; pr = G.tok[pr].sib) {
        int mi = jint(&G, jget(&G, pr, "material"), 0);
        if (mi < 0 || mi >= g_nmats) mi = 0;
        int attr = jget(&G, pr, "attributes");
        AccView pos = acc_view(&G, jint(&G, jget(&G, attr, "POSITION"), 0), 3);
        int uvIdx = jint(&G, jget(&G, attr, "TEXCOORD_0"), -1);
        int nrmIdx = jint(&G, jget(&G, attr, "NORMAL"), -1);
        AccView uv = { 0 }, nrm = { 0 };
        if (uvIdx >= 0) uv = acc_view(&G, uvIdx, 2);
        if (nrmIdx >= 0) nrm = acc_view(&G, nrmIdx, 3);
        int base = g_nverts;
        for (int i = 0; i < pos.count; i++) {
            Vertex* v = &g_verts[g_nverts++];
            const float* P = (const float*)(pos.p + (size_t)i * pos.stride);
            v->px = P[0] * scl[0] + trn[0];
            v->py = P[1] * scl[1] + trn[1];
            v->pz = P[2] * scl[2] + trn[2];
            if (uv.p) { const float* T = (const float*)(uv.p + (size_t)i * uv.stride); v->u = T[0]; v->v = T[1]; }
            else v->u = v->v = 0.0f;
            if (nrm.p) { const float* N = (const float*)(nrm.p + (size_t)i * nrm.stride); v->nx = N[0]; v->ny = N[1]; v->nz = N[2]; }
            else { v->nx = 0.0f; v->ny = 1.0f; v->nz = 0.0f; }
        }
        AccView ix = acc_view(&G, jint(&G, jget(&G, pr, "indices"), 0), 1);
        for (int t = 0; t < ix.count / 3; t++) {
            uint32_t i0, i1, i2;
            if (ix.comp == 5123) {
                const uint16_t* s = (const uint16_t*)(ix.p + (size_t)t * 3 * ix.stride);
                i0 = s[0]; i1 = s[1]; i2 = s[2];
            } else {
                const uint32_t* s = (const uint32_t*)(ix.p + (size_t)t * 3 * ix.stride);
                i0 = s[0]; i1 = s[1]; i2 = s[2];
            }
            g_idx[g_ntris * 3 + 0] = base + i0;          /* the winding swap: i2 before i1 */
            g_idx[g_ntris * 3 + 1] = base + i2;
            g_idx[g_ntris * 3 + 2] = base + i1;
            g_triMat[g_ntris] = (uint16_t)mi;
            g_triCullBack[g_ntris] = !twoSided[mi];
            g_triNoCast[g_ntris] = (uint8_t)g_mats[mi].alphaMask;
            g_ntris++;
        }
    }
    free(G.tok); free(G.json); free(G.bin);

    g_aabbMin = v3(1e30f, 1e30f, 1e30f); g_aabbMax = v3(-1e30f, -1e30f, -1e30f);
    for (int i = 0; i < g_nverts; i++) {
        const Vertex* v = &g_verts[i];
        if (v->px < g_aabbMin.x) g_aabbMin.x = v->px;
        if (v->py < g_aabbMin.y) g_aabbMin.y = v->py;
        if (v->pz < g_aabbMin.z) g_aabbMin.z = v->pz;
        if (v->px > g_aabbMax.x) g_aabbMax.x = v->px;
        if (v->py > g_aabbMax.y) g_aabbMax.y = v->py;
        if (v->pz > g_aabbMax.z) g_aabbMax.z = v->pz;
    }

    /* SoA positions for the 8-wide transforms, and the per-frame buffers,
       padded to 8 vertices so the last block full-stores */
    int nv8 = (g_nverts + 7) & ~7;
    g_vpx = (float*)SDL_aligned_alloc(32, (size_t)nv8 * 4);
    g_vpy = (float*)SDL_aligned_alloc(32, (size_t)nv8 * 4);
    g_vpz = (float*)SDL_aligned_alloc(32, (size_t)nv8 * 4);
    for (int i = 0; i < nv8; i++) {
        const int in = i < g_nverts;
        g_vpx[i] = in ? g_verts[i].px : 0.0f; g_vpy[i] = in ? g_verts[i].py : 0.0f; g_vpz[i] = in ? g_verts[i].pz : 0.0f;
    }
    g_clip = (V4*)SDL_aligned_alloc(32, (size_t)nv8 * sizeof(V4));
    g_scr = (ScrVert*)SDL_aligned_alloc(32, (size_t)nv8 * sizeof(ScrVert));
    g_oc = (uint8_t*)SDL_aligned_alloc(32, (size_t)nv8);
    g_triCap = g_ntris * 2 + 4096;                       /* near-plane clipping fans one triangle into several */
    g_tris = (Tri*)SDL_aligned_alloc(64, (size_t)g_triCap * sizeof(Tri));
    g_ntriChunks = (g_ntris + TRI_CHUNK - 1) >> TRI_CHUNK_SHIFT;
}

/* ============================================================================
   The cloud texture (R13): tileable 5-octave value noise, then a coverage window
   ============================================================================ */
static float noise_hash(int x, int y, int oct)
{
    uint32_t h = (uint32_t)(x * 374761393 + y * 668265263 + oct * 2246822519u);
    h ^= h >> 13; h *= 1274126177u; h ^= h >> 16;
    return (float)(h & 0xFFFF) * (1.0f / 65535.0f);
}
static float smooth01(float t) { return t * t * (3.0f - 2.0f * t); }

static void bake_clouds(void)
{
    g_cloudDensity = (uint8_t*)SDL_aligned_alloc(64, CLOUD_RES * CLOUD_RES + 64);
    for (int y = 0; y < CLOUD_RES; y++)
        for (int x = 0; x < CLOUD_RES; x++) {
            float sum = 0.0f, amp = 1.0f, norm = 0.0f;
            for (int o = 0; o < 5; o++) {
                int freq = 4 << o;
                float fx = (float)x * freq / CLOUD_RES, fy = (float)y * freq / CLOUD_RES;
                int ix = (int)fx, iy = (int)fy;
                float tx = smooth01(fx - ix), ty = smooth01(fy - iy);
                int x1 = (ix + 1) & (freq - 1), y1 = (iy + 1) & (freq - 1);
                float n00 = noise_hash(ix, iy, o), n10 = noise_hash(x1, iy, o);
                float n01 = noise_hash(ix, y1, o), n11 = noise_hash(x1, y1, o);
                float n = (n00 * (1 - tx) + n10 * tx) * (1 - ty) + (n01 * (1 - tx) + n11 * tx) * ty;
                sum += n * amp; norm += amp; amp *= 0.5f;
            }
            float d = clampf((sum / norm - SKY_cloudCoverLo) / (SKY_cloudCoverHi - SKY_cloudCoverLo), 0.0f, 1.0f);
            g_cloudDensity[y * CLOUD_RES + x] = (uint8_t)(smooth01(d) * 255.0f + 0.5f);
        }
    memset(g_cloudDensity + CLOUD_RES * CLOUD_RES, 0, 64);
}
