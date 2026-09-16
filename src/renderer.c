/* renderer.c -- the renderer's single translation unit, compiled with AVX2 + FMA code generation (see CMakeLists.txt).
   The order is the dependency order: pool, loading, the shadow map (its
   sampler is inlined into the shader), geometry, the tile raster, the app. */
#include "core.h"
#include "pool.c"
#include "scene.c"
#include "shadow.c"
#include "geometry.c"
#include "raster.c"
#include "app.c"
