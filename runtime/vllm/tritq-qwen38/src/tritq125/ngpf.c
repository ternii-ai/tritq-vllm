/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Wai Chung */
/* ngpf.c (n-gram lookup): CPU-side work on the PLE n-gram row ids, run IN STREAM ORDER on the lookup side stream as a CUDA host
 * function (cuLaunchHostFunc; captured as a host node inside CUDA graphs), between the D2H copy of the step's ids and the GPU gather.
 *  - prefetch (do_pf): posix_fadvise(WILLNEED) on every page of the step's rows (asynchronous readahead: all misses go to the NVMe at once
 *    instead of one GPU fault at a time), then one read per page through the SAME mapping the GPU dereferences (waits for the reads, makes
 *    the PTEs present) -> the gather finds resident, mapped pages. Reads only; never changes the gathered bytes.
 *  - stats (do_stat): mincore() per page BEFORE prefetching -> pages not resident = what the GPU would have faulted from the NVMe.
 *  - dump (dumpfd >= 0): append int32 T, int32 heads, T*heads int32 ids (the TQ_NG_DUMP record format) - works inside CUDA graphs.
 * Slots: userData = slot_t {ids (page-locked host buffer, cuMemHostRegister), n, ctx}, all allocated at init (before graph capture);
 * the caller uses a ring for eager lookups and one permanent slot per captured lookup. The D2H copy is issued here too (cuMemcpyDtoHAsync
 * on the same stream, captured as a memcpy node), so no torch host allocator is involved.
 * No CUDA headers: the driver entry points are resolved from libcuda.so.1. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

typedef struct { int64_t *ids; volatile int n; void *ctx; } slot_t;
typedef struct { int fd, dumpfd, heads, do_pf, do_stat; const volatile uint8_t *base; int64_t nrows, rb; int64_t *pg; int32_t *dbuf; int maxn;
                 int (*launch)(void *, void (*)(void *), void *); int (*d2h)(void *, uint64_t, size_t, void *); slot_t *slots; int nslots;
                 volatile int64_t calls, ids, pages, nonres, sink; } ctx_t;

static int cmp64(const void *a, const void *b) { int64_t x = *(const int64_t *)a, y = *(const int64_t *)b; return (x > y) - (x < y); }

static void cb(void *p) {
  slot_t *s = (slot_t *)p; ctx_t *c = (ctx_t *)s->ctx; int n = s->n, m = 0; int64_t *pg = c->pg;
  if (c->dumpfd >= 0) {
    int32_t h[2] = {n / c->heads, c->heads}; for (int k = 0; k < n; k++) c->dbuf[k] = (int32_t)s->ids[k];
    if (write(c->dumpfd, h, 8) != 8 || write(c->dumpfd, c->dbuf, 4 * (size_t)n) != 4 * (ssize_t)n) c->dumpfd = -1;
  }
  if (!c->do_pf && !c->do_stat) { c->calls++; c->ids += n; return; }
  for (int k = 0; k < n; k++) {
    int64_t i = s->ids[k]; if (i < 0) i = 0; if (i >= c->nrows) i = c->nrows - 1;
    int64_t a = i * c->rb; pg[m++] = a >> 12; pg[m++] = (a + c->rb - 1) >> 12;
  }
  qsort(pg, m, sizeof(int64_t), cmp64);
  int u = 0; for (int k = 0; k < m; k++) if (!u || pg[k] != pg[u - 1]) pg[u++] = pg[k];
  if (c->do_stat) { unsigned char v; int64_t nr = 0;
    for (int k = 0; k < u; k++) if (mincore((void *)(c->base + (pg[k] << 12)), 4096, &v) == 0 && !(v & 1)) nr++;
    c->nonres += nr; }
  if (c->do_pf) {
    for (int k = 0; k < u; k++) posix_fadvise(c->fd, pg[k] << 12, 4096, POSIX_FADV_WILLNEED);
    int64_t acc = 0; for (int k = 0; k < u; k++) acc += c->base[pg[k] << 12];
    c->sink += acc;
  }
  c->calls++; c->ids += n; c->pages += u;
  if (c->calls % 1000 == 0)
    fprintf(stderr, "tritq n-gram: ngpf calls %lld ids %lld pages %lld nonresident %lld (pf %d stat %d)\n", (long long)c->calls,
            (long long)c->ids, (long long)c->pages, (long long)c->nonres, c->do_pf, c->do_stat);
}

void *ngpf_init(const char *path, uint64_t base, int64_t nrows, int64_t rb, int maxn, int heads, int do_pf, int do_stat, const char *dump,
                int nslots) {
  void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL); if (!h) return 0;
  ctx_t *c = calloc(1, sizeof(ctx_t)); c->launch = dlsym(h, "cuLaunchHostFunc"); c->d2h = dlsym(h, "cuMemcpyDtoHAsync_v2");
  int (*reg)(void *, size_t, unsigned) = dlsym(h, "cuMemHostRegister_v2"); if (!c->launch || !c->d2h || !reg) return 0;
  c->fd = open(path, O_RDONLY); if (c->fd < 0) return 0;
  c->dumpfd = (dump && *dump) ? open(dump, O_WRONLY | O_CREAT | O_APPEND, 0644) : -1;
  c->base = (const volatile uint8_t *)base; c->nrows = nrows; c->rb = rb; c->maxn = maxn; c->heads = heads; c->do_pf = do_pf; c->do_stat = do_stat;
  c->pg = malloc(sizeof(int64_t) * 2 * (size_t)maxn); c->dbuf = malloc(sizeof(int32_t) * (size_t)maxn);
  c->nslots = nslots; c->slots = calloc(nslots, sizeof(slot_t));
  for (int k = 0; k < nslots; k++) {
    size_t b = ((sizeof(int64_t) * (size_t)maxn) + 4095) & ~(size_t)4095; void *q = 0;
    if (posix_memalign(&q, 4096, b) || reg(q, b, 0)) return 0;
    c->slots[k].ids = q; c->slots[k].ctx = c;
  }
  return c;
}
/* enqueue on stream: D2H copy of n int64 ids from device address dev into slot k, then the host function on slot k */
int ngpf_launch(void *cv, void *stream, int k, uint64_t dev, int n) {
  ctx_t *c = cv; if (k < 0 || k >= c->nslots || n > c->maxn) return -1; c->slots[k].n = n;
  int r = c->d2h(c->slots[k].ids, dev, sizeof(int64_t) * (size_t)n, stream); if (r) return r;
  return c->launch(stream, cb, &c->slots[k]);
}
void ngpf_stats(void *cv, int64_t *o) { ctx_t *c = cv; o[0] = c->calls; o[1] = c->ids; o[2] = c->pages; o[3] = c->nonres; }
