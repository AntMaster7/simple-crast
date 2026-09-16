/* cpucheck.c -- the entry point, compiled WITHOUT AVX2 code generation.

   Everything in renderer.c is built for AVX2 + FMA, and the compiler is free
   to use VEX-encoded instructions anywhere in it -- including code that runs
   before any check could. This file stays plain x86-64 (SSE2), confirms the
   CPU and the OS support AVX2 and FMA3, and only then calls into the AVX2
   code. SDL_main.h turns main() into the platform's entry point (WinMain on
   Windows). */
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

int renderer_main(void);

static bool cpu_has_fma(void)
{
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 1);
    return (r[2] >> 12) & 1;
#else
    unsigned a, b, c, d;
    return __get_cpuid(1, &a, &b, &c, &d) && ((c >> 12) & 1);
#endif
}

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;
    if (!SDL_HasAVX2() || !cpu_has_fma()) {   /* SDL_HasAVX2 also checks that the OS saves the YMM state */
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Sponza (simple)",
                                 "This renderer needs a CPU with AVX2 and FMA3\n(Intel Haswell / AMD Zen or newer).", NULL);
        return 1;
    }
    return renderer_main();
}
