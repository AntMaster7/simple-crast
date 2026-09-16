/* pool.c -- the thread pool: NUM_THREADS threads (the main thread plus
   workers) race on one atomic job counter. dispatch() runs a phase to
   completion; the main thread works too.

   Per frame there are three phases -- PH_VERTS (vertex transform), PH_TRIS
   (triangle setup + tile binning) and PH_TILES (one 64x64 tile end to end) --
   plus four load-time ones (normal maps and the shadow bake). */
#include "core.h"

static SDL_Semaphore* g_wake;         /* one signal per worker per phase */
static SDL_Semaphore* g_done;         /* signalled by the last thread out of a phase */
static SDL_AtomicInt g_quitWorkers, g_jobNext, g_activeThreads;
static int g_phase, g_jobTotal;       /* set before the workers are woken */

static void run_jobs(int tid)
{
    for (;;) {
        const int j = SDL_AddAtomicInt(&g_jobNext, 1);   /* claim the next job (returns the value before the add) */
        if (j >= g_jobTotal) break;
        switch (g_phase) {
        case PH_VERTS:   job_verts(j); break;
        case PH_TRIS:    job_tris(j, tid); break;         /* tid picks the thread's Tri slot cache */
        case PH_TILES:   job_tile(g_tileOrder[j]); break; /* costliest tiles first */
        case PH_NRM:     job_nrm(j); break;
        case PH_SMVERTS: job_sm_verts(j); break;
        case PH_SMSETUP: job_sm_setup(j); break;
        case PH_SMTILES: job_sm_tile(j); break;
        }
    }
    if (SDL_AddAtomicInt(&g_activeThreads, -1) == 1) SDL_SignalSemaphore(g_done);   /* last one out */
}

static int SDLCALL worker_proc(void* param)
{
    const int tid = (int)(intptr_t)param;
    for (;;) {
        SDL_WaitSemaphore(g_wake);
        if (SDL_GetAtomicInt(&g_quitWorkers)) return 0;
        run_jobs(tid);
    }
}

static void dispatch(int phase, int total)
{
    g_phase = phase;
    g_jobTotal = total;
    SDL_SetAtomicInt(&g_jobNext, 0);
    SDL_SetAtomicInt(&g_activeThreads, NUM_THREADS);
    for (int i = 1; i < NUM_THREADS; i++) SDL_SignalSemaphore(g_wake);   /* wake every worker */
    run_jobs(0);
    SDL_WaitSemaphore(g_done);
}

static void pool_init(void)
{
    g_wake = SDL_CreateSemaphore(0);
    g_done = SDL_CreateSemaphore(0);
    for (int i = 1; i < NUM_THREADS; i++)
        SDL_DetachThread(SDL_CreateThread(worker_proc, "worker", (void*)(intptr_t)i));
}

static void pool_shutdown(void)
{
    SDL_SetAtomicInt(&g_quitWorkers, 1);
    for (int i = 1; i < NUM_THREADS; i++) SDL_SignalSemaphore(g_wake);
}
