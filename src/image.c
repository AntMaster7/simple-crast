/* image.c -- the JPEG / PNG decoder: stb_image (public domain), compiled once.

   The header is the copy that ships inside the SDL3 source tree
   (SDL3/src/video/stb_image.h), so the renderer needs no library beyond SDL.
   SDL compiles its own copy with STB_IMAGE_STATIC, so the two never clash at
   link time. SDL's copy is edited to use SDL's integer types (Uint8, ...),
   hence the SDL include first. */
#include <SDL3/SDL.h>
#include <stdlib.h>                   /* SDL's copy dropped this include; the decoder calls malloc / realloc / free */
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO                 /* SDL's copy only keeps the in-memory decoder intact */
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#ifdef _MSC_VER
#pragma warning(disable: 4244 4456 4457 4701)   /* the decoder's own conversions and shadowing */
#endif
#include "stb_image.h"
