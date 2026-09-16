/* app.c -- the SDL3 window, the fly-through camera (R1), the FPS counter and
   the frame loop.

   A frame: camera -> cluster cull flags -> PH_VERTS -> PH_TRIS -> order the
   tiles costliest first -> PH_TILES -> FPS overlay -> one software blit into
   the window surface. There is no temporal state in the image, so the counter
   can be drawn straight into the frame buffer before the present. */
#include "core.h"

static SDL_Window* g_window;
static SDL_Surface* g_frameSurface;   /* g_frame wrapped for SDL_BlitSurface, which converts to the window's format */
static int g_running = 1, g_resizePending = 1;
static int g_mouseLook;
static float g_lookDX, g_lookDY;      /* relative mouse motion gathered from this frame's events */

/* ============================================================================
   Frame targets: the frame buffer, bins, chunk bitmaps and tile order for a
   window size
   ============================================================================ */
static void targets_create(int w, int h)
{
    if (g_bins) {
        for (size_t i = 0, n = (size_t)g_ntriChunks * g_ntiles; i < n; i++) free(g_bins[i].items);
        free(g_bins); free(g_tileChunkBits); free(g_tileOrder); free(g_tileCost);
        SDL_DestroySurface(g_frameSurface);
        SDL_aligned_free(g_frame);
    }
    g_width = w; g_height = h;
    g_halfWf = (float)w * 0.5f; g_halfHf = (float)h * 0.5f;
    g_tilesX = (w + TILE_SIZE - 1) >> TILE_SHIFT;
    g_tilesY = (h + TILE_SIZE - 1) >> TILE_SHIFT;
    g_ntiles = g_tilesX * g_tilesY;
    g_frameStride = g_tilesX * TILE_SIZE;
    g_frame = (uint32_t*)SDL_aligned_alloc(32, (size_t)g_frameStride * h * 4);
    g_bins = (Bin*)calloc((size_t)g_ntriChunks * g_ntiles, sizeof(Bin));   /* stamp 0: every bin reads empty */
    g_chunkWords = (g_ntriChunks + 31) >> 5;
    g_tileChunkBits = (SDL_AtomicInt*)calloc((size_t)g_ntiles * g_chunkWords, sizeof(SDL_AtomicInt));
    g_tileOrder = (int*)malloc((size_t)g_ntiles * sizeof(int));
    g_tileCost = (unsigned*)calloc((size_t)g_ntiles, sizeof(unsigned));
    for (int i = 0; i < g_ntiles; i++) g_tileOrder[i] = i;
    /* 0xFFRRGGBB in native byte order is SDL's XRGB8888; the pitch is the
       whole-tile stride, the width the window's */
    g_frameSurface = SDL_CreateSurfaceFrom(w, h, SDL_PIXELFORMAT_XRGB8888, g_frame, g_frameStride * 4);
}

/* ============================================================================
   Camera (R1): left-handed basis, no roll, 70 degrees vertical
   ============================================================================ */
static void camera_frame_constants(void)
{
    const float cp = cosf(g_camPitch), sp = sinf(g_camPitch), cy = cosf(g_camYaw), sy = sinf(g_camYaw);
    g_fwd = v3(cp * sy, sp, cp * cy);                         /* yaw 0 faces +Z */
    g_rightv = v3_norm(v3_cross(v3(0, 1, 0), g_fwd));
    g_upv = v3_cross(g_fwd, g_rightv);
    const float tanHalf = tanf(CAM_FOV_DEG * 3.14159265f / 360.0f);
    const float ct = 1.0f / tanHalf, ctx = ct / (g_halfWf / g_halfHf);
    const float A = CAM_ZFAR / (CAM_ZFAR - CAM_ZNEAR), B = -CAM_ZNEAR * A;
    const V3 R = g_rightv, U = g_upv, F = g_fwd, P = g_camPos;
    const float rw = -v3_dot(R, P), uw = -v3_dot(U, P), fw = -v3_dot(F, P);
    g_vp.m[0][0] = R.x * ctx; g_vp.m[0][1] = R.y * ctx; g_vp.m[0][2] = R.z * ctx; g_vp.m[0][3] = rw * ctx;
    g_vp.m[1][0] = U.x * ct;  g_vp.m[1][1] = U.y * ct;  g_vp.m[1][2] = U.z * ct;  g_vp.m[1][3] = uw * ct;
    g_vp.m[2][0] = F.x * A;   g_vp.m[2][1] = F.y * A;   g_vp.m[2][2] = F.z * A;   g_vp.m[2][3] = fw * A + B;
    g_vp.m[3][0] = F.x;       g_vp.m[3][1] = F.y;       g_vp.m[3][2] = F.z;       g_vp.m[3][3] = fw;
    const float s = 2.0f * tanHalf / (2.0f * g_halfHf);       /* one pixel, for the sky rays */
    g_rayB = v3_scale(g_rightv, s);
    g_rayC = v3_scale(g_upv, -s);                             /* y-down */
    g_rayA = v3_add(g_fwd, v3_add(v3_scale(g_rightv, -s * g_halfWf), v3_scale(g_upv, s * g_halfHf)));
}

static void camera_update(float dt)
{
    const float lookDX = g_lookDX, lookDY = g_lookDY;
    g_lookDX = g_lookDY = 0.0f;                               /* consumed this frame either way */
    if (!(SDL_GetWindowFlags(g_window) & SDL_WINDOW_INPUT_FOCUS)) return;
    const bool* key = SDL_GetKeyboardState(NULL);
    float sp = g_moveSpeed * dt;
    if (key[SDL_SCANCODE_LSHIFT] || key[SDL_SCANCODE_RSHIFT]) sp *= 4.0f;
    if (key[SDL_SCANCODE_LCTRL] || key[SDL_SCANCODE_RCTRL]) sp *= 0.25f;
    V3 d = v3(0, 0, 0);
    if (key[SDL_SCANCODE_W]) d = v3_add(d, g_fwd);
    if (key[SDL_SCANCODE_S]) d = v3_sub(d, g_fwd);
    if (key[SDL_SCANCODE_D]) d = v3_add(d, g_rightv);
    if (key[SDL_SCANCODE_A]) d = v3_sub(d, g_rightv);
    if (key[SDL_SCANCODE_E]) d = v3_add(d, v3(0, 1, 0));
    if (key[SDL_SCANCODE_Q]) d = v3_sub(d, v3(0, 1, 0));
    g_camPos = v3_add(g_camPos, v3_scale(d, sp));
    const float turn = 1.8f * dt;
    if (key[SDL_SCANCODE_LEFT])  g_camYaw -= turn;
    if (key[SDL_SCANCODE_RIGHT]) g_camYaw += turn;
    if (key[SDL_SCANCODE_UP])    g_camPitch += turn;
    if (key[SDL_SCANCODE_DOWN])  g_camPitch -= turn;
    if (g_mouseLook) {                                        /* right mouse held: relative mode, the motion turns */
        g_camYaw += lookDX * 0.0032f;
        g_camPitch -= lookDY * 0.0032f;
    }
    g_camPitch = clampf(g_camPitch, -1.53f, 1.53f);
}

/* ============================================================================
   The FPS counter: a 5x7 bitmap font drawn into the top-left of the frame
   ============================================================================ */
static const struct { char c; unsigned char rows[7]; } GLYPHS[] = {
    { '0', { 0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E } }, { '1', { 0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E } },
    { '2', { 0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F } }, { '3', { 0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E } },
    { '4', { 0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02 } }, { '5', { 0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E } },
    { '6', { 0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E } }, { '7', { 0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08 } },
    { '8', { 0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E } }, { '9', { 0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C } },
    { '.', { 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C } }, { 'F', { 0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10 } },
    { 'P', { 0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10 } }, { 'S', { 0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E } },
    { 'm', { 0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11 } }, { 's', { 0x00, 0x00, 0x0E, 0x10, 0x0E, 0x01, 0x1E } },
};

static void draw_text(int x0, int y0, const char* text, int scale)
{
    const int len = (int)strlen(text), cw = 6 * scale, pad = 2 * scale;
    const int bx1 = x0 + len * cw + pad, by1 = y0 + 7 * scale + pad;
    for (int y = y0 - pad; y < by1; y++)                      /* a darkened box behind the text */
        for (int x = x0 - pad; x < bx1; x++)
            if (x >= 0 && y >= 0 && x < g_width && y < g_height) {
                uint32_t* p = &g_frame[(size_t)y * g_frameStride + x];
                *p = 0xFF000000u | ((*p >> 2) & 0x3F3F3F);
            }
    for (int i = 0; i < len; i++) {
        const unsigned char* rows = NULL;
        for (int g = 0; g < (int)(sizeof GLYPHS / sizeof GLYPHS[0]); g++)
            if (GLYPHS[g].c == text[i]) rows = GLYPHS[g].rows;
        if (!rows) continue;                                  /* space */
        for (int r = 0; r < 7 * scale; r++)
            for (int c = 0; c < 5 * scale; c++)
                if ((rows[r / scale] >> (4 - c / scale)) & 1) {
                    const int x = x0 + i * cw + c, y = y0 + r;
                    if (x < g_width && y < g_height) g_frame[(size_t)y * g_frameStride + x] = 0xFFFFFFFFu;
                }
    }
}

/* The frame time the counter shows, in ms, fed every frame with this frame's
   wall time and the running clock in seconds. The shown value is the mean
   frame time over the last FPS_WINDOW_SEC and changes only once per window,
   so the digits hold still long enough to read. */
#define FPS_WINDOW_SEC 0.5
static double fps_display_ms(double frameMs, double nowSec)
{
    static double windowStart, sumMs, shownMs;
    static int frames;
    sumMs += frameMs;
    frames++;
    if (shownMs == 0.0) shownMs = frameMs;               /* something to show before the first window closes */
    if (nowSec - windowStart >= FPS_WINDOW_SEC) {
        shownMs = sumMs / frames;
        windowStart = nowSec; sumMs = 0.0; frames = 0;
    }
    return shownMs;
}

/* ============================================================================
   Window
   ============================================================================ */
static void mouse_look(int on)
{
    if (on == g_mouseLook) return;
    g_mouseLook = on;
    SDL_SetWindowRelativeMouseMode(g_window, on != 0);        /* hidden cursor, motion never stops at an edge */
}

static void handle_events(void)
{
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        switch (e.type) {
        case SDL_EVENT_QUIT:              g_running = 0; break;
        case SDL_EVENT_KEY_DOWN:          if (e.key.scancode == SDL_SCANCODE_ESCAPE) g_running = 0; break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN: if (e.button.button == SDL_BUTTON_RIGHT) mouse_look(1); break;
        case SDL_EVENT_MOUSE_BUTTON_UP:   if (e.button.button == SDL_BUTTON_RIGHT) mouse_look(0); break;
        case SDL_EVENT_MOUSE_MOTION:      if (g_mouseLook) { g_lookDX += e.motion.xrel; g_lookDY += e.motion.yrel; } break;
        case SDL_EVENT_WINDOW_FOCUS_LOST: mouse_look(0); break;
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: g_resizePending = 1; break;
        }
    }
}

static int file_exists(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (f) fclose(f);
    return f != NULL;
}

/* the exe's directory, or up to three of its parents (a build tree inside the repo) */
static void find_asset_dir(void)
{
    const char* base = SDL_GetBasePath();
    char dir[PATH_LEN];
    snprintf(dir, sizeof dir, "%s", base ? base : "./");
    for (int up = 0; up < 4; up++) {
        char probe[PATH_LEN + 32];
        snprintf(probe, sizeof probe, "%s" SCENE_DIR "Sponza.gltf", dir);
        if (file_exists(probe)) { snprintf(g_assetDir, sizeof g_assetDir, "%s", dir); return; }
        size_t n = strlen(dir);
        if (n < 2) break;
        dir[n - 1] = 0;                                        /* drop the trailing separator ... */
        char* sep = strrchr(dir, '/');
        char* bsl = strrchr(dir, '\\');
        if (bsl > sep) sep = bsl;
        if (!sep) break;
        sep[1] = 0;                                            /* ... and the last component */
    }
    snprintf(g_assetDir, sizeof g_assetDir, "%s", base ? base : "./");
}

/* ============================================================================
   Entry (called from cpucheck.c once AVX2 is confirmed)
   ============================================================================ */
int renderer_main(void)
{
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Sponza (simple)", SDL_GetError(), NULL);
        return 1;
    }
    for (int b = 0; b < 256; b++)
        for (int l = 0; l < 8; l++) g_maskLut[b][l] = (b >> l) & 1 ? -1 : 0;
    find_asset_dir();
    pool_init();

    /* ---- loading ---- */
    load_gltf();
    clusters_build();
    bake_clouds();
    g_sunDir = v3_norm(v3(LIT_sunDir[0], LIT_sunDir[1], LIT_sunDir[2]));   /* the fixed noon sun */
    shadow_bake();
    g_cloudPlaneY = g_aabbMax.y + SKY_cloudHeight;
    {   /* the sun disc rim as cosines (R13): cos t ~ 1 - t^2/2 + t^4/24 for these small angles */
        const double d2r = 3.14159265358979323846 / 180.0;
        const double to = (SKY_sunRadiusDeg + SKY_sunEdgeDeg) * d2r, ti = SKY_sunRadiusDeg * d2r;
        const double co = 1.0 - to * to / 2.0 + to * to * to * to / 24.0;
        const double ci = 1.0 - ti * ti / 2.0 + ti * ti * ti * ti / 24.0;
        g_sunCosOuter = (float)co;
        g_sunEdgeInv = (float)(1.0 / (ci - co));
    }

    {   /* start at one end of the atrium at eye height, looking down its length */
        const V3 c = v3_scale(v3_add(g_aabbMin, g_aabbMax), 0.5f), ext = v3_sub(g_aabbMax, g_aabbMin);
        g_moveSpeed = sqrtf(v3_dot(ext, ext)) * 0.12f;
        g_camPos = v3(g_aabbMin.x + ext.x * 0.82f, g_aabbMin.y + ext.y * 0.22f, c.z);
        g_camYaw = -1.5707963f;
        g_camPitch = 0.06f;
    }

    /* ---- the window, rendered at its full pixel size ---- */
    g_window = SDL_CreateWindow("Sponza -- simple AVX2 software renderer", 1600, 900,
                                SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!g_window) {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Sponza (simple)", SDL_GetError(), NULL);
        return 1;
    }

    const Uint64 freq = SDL_GetPerformanceFrequency();
    Uint64 prev = SDL_GetPerformanceCounter();
    const Uint64 start = prev;
    while (g_running) {
        handle_events();
        if (!g_running) break;
        if (g_resizePending) {
            g_resizePending = 0;
            int w, h;
            SDL_GetWindowSizeInPixels(g_window, &w, &h);
            if (w > 0 && h > 0 && (w != g_width || h != g_height)) targets_create(w, h);
        }
        if ((SDL_GetWindowFlags(g_window) & SDL_WINDOW_MINIMIZED) || !g_frame) { SDL_Delay(10); continue; }

        const Uint64 now = SDL_GetPerformanceCounter();
        const double dt = (double)(now - prev) / (double)freq;
        prev = now;
        camera_update((float)(dt > 0.1 ? 0.1 : dt));
        camera_frame_constants();
        clusters_classify();

        SDL_SetAtomicInt(&g_triCount, 0);                    /* reset the Tri allocator ... */
        for (int i = 0; i < NUM_THREADS; i++) { g_slot[i].next = 0; g_slot[i].end = 0; }
        g_frameStamp++;                                      /* ... empty every bin ... */
        memset(g_tileChunkBits, 0, (size_t)g_ntiles * g_chunkWords * sizeof(SDL_AtomicInt));   /* ... and the chunk bitmaps */

        dispatch(PH_VERTS, (g_nverts + VERT_CHUNK - 1) / VERT_CHUNK);
        dispatch(PH_TRIS, g_ntriChunks);
        for (int i = 1; i < g_ntiles; i++) {                 /* costliest tile first (nearly sorted already) */
            const int t = g_tileOrder[i];
            const unsigned c = g_tileCost[t];
            int j = i - 1;
            while (j >= 0 && g_tileCost[g_tileOrder[j]] < c) { g_tileOrder[j + 1] = g_tileOrder[j]; j--; }
            g_tileOrder[j + 1] = t;
        }
        dispatch(PH_TILES, g_ntiles);

        {
            const double ms = fps_display_ms(dt * 1000.0, (double)(now - start) / (double)freq);
            char text[64];
            snprintf(text, sizeof text, "%.0f FPS  %.2f ms", ms > 0.0 ? 1000.0 / ms : 0.0, ms);
            draw_text(10, 10, text, 2);
        }
        SDL_Surface* ws = SDL_GetWindowSurface(g_window);
        if (ws) {
            SDL_BlitSurface(g_frameSurface, NULL, ws, NULL);
            SDL_UpdateWindowSurface(g_window);
        }
    }

    mouse_look(0);
    pool_shutdown();
    SDL_DestroyWindow(g_window);
    SDL_Quit();
    return 0;
}
