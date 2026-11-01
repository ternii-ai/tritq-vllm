# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wai Chung
"""bench_oai.py: engine-neutral decode-speed benchmark over the OpenAI-compatible streaming chat API, so vLLM (NVFP4, with or
without MTP / DFlash) and llama-server (our exact trit GGUFs) are measured by the SAME client, prompts and settings.
Per request: TTFT (send -> first content token) and decode tok/s = (completion_tokens - 1) / (t_last - t_first), completion_tokens from the
server's usage block (stream_options.include_usage). Greedy, non-thinking (chat_template_kwargs enable_thinking=false; llama-server uses its
nothink template), natural stop, max_tokens MAXTOK. Concurrency C: C requests in flight at once over the prompt list (each prompt run C/len
times as needed); aggregate = total completion tokens / wall time of the round.
usage: bench_oai.py <base url e.g. http://127.0.0.1:8000> <model name> <label> [C list, default 1,4,8] [repeats, default 2]
Prints one JSON line per C to stdout (append to a results file)."""
import sys, json, time, threading, http.client, urllib.parse, statistics

PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Explain step by step.",
    "Write a Python function `merge_intervals(intervals)` that merges overlapping intervals, with docstring, type hints and three doctest examples.",
    "Explain how a hash map works, including collision handling, load factor and resizing, in about 300 words.",
    "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
    "A train leaves at 9:40 and travels 270 km at 90 km/h, then waits 25 minutes, then travels 150 km at 100 km/h. When does it arrive? Show your working.",
    "Write a Python class implementing an LRU cache with get and put in O(1), and explain the design.",
    "Summarise the causes and consequences of the 2008 financial crisis in five paragraphs.",
    "Write a bash script that finds the 10 largest files under a directory, prints their sizes in human-readable form, and handles spaces in names.",
]


def one(base, model, prompt, maxtok, out):
    u = urllib.parse.urlparse(base); c = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": maxtok, "temperature": 0, "top_p": 1,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter(); c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse(); tf = tl = None; ntok = None; nchunks = 0; buf = b""
    while True:
        line = r.readline()
        if not line: break
        line = line.strip()
        if not line.startswith(b"data:"): continue
        d = line[5:].strip()
        if d == b"[DONE]": break
        j = json.loads(d)
        if j.get("usage"): ntok = j["usage"].get("completion_tokens", ntok)
        for ch in j.get("choices") or []:
            delta = ch.get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                now = time.perf_counter(); tf = tf or now; tl = now; nchunks += 1
    c.close()
    out.append({"ttft": (tf - t0) if tf else None, "ntok": ntok, "chunks": nchunks,
                "dec": (ntok - 1) / (tl - tf) if (tf and tl and ntok and tl > tf) else None, "wall": time.perf_counter() - t0})


def round_(base, model, C, maxtok):
    res = []; ths = []; t0 = time.perf_counter()
    jobs = [PROMPTS[i % len(PROMPTS)] for i in range(max(C, len(PROMPTS)) if C > 1 else len(PROMPTS))]
    # C=1: the 8 prompts one after another; C>1: waves of C concurrent requests
    for w in range(0, len(jobs), C):
        ths = [threading.Thread(target=one, args=(base, model, p, maxtok, res)) for p in jobs[w:w + C]]
        [t.start() for t in ths]; [t.join() for t in ths]
    wall = time.perf_counter() - t0
    return res, wall


if __name__ == "__main__":
    base, model, label = sys.argv[1:4]
    Cs = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1,4,8").split(",")]
    reps = int(sys.argv[5]) if len(sys.argv) > 5 else 2; maxtok = 512
    one(base, model, "Say hello.", 16, [])                                   # warm-up
    for C in Cs:
        rows = []; walls = []; toks = 0
        for _ in range(reps):
            r, w = round_(base, model, C, maxtok); rows += r; walls.append(w); toks += sum(x["ntok"] or 0 for x in r)
        dec = [x["dec"] for x in rows if x["dec"]]; ttft = [x["ttft"] for x in rows if x["ttft"]]
        print(json.dumps({"label": label, "C": C, "requests": len(rows), "mean_out_tokens": round(statistics.mean(x["ntok"] or 0 for x in rows), 1),
                          "decode_tok_s_per_stream_mean": round(statistics.mean(dec), 2), "decode_median": round(statistics.median(dec), 2),
                          "decode_min": round(min(dec), 2), "decode_max": round(max(dec), 2), "ttft_ms_median": round(1000 * statistics.median(ttft), 1),
                          "aggregate_tok_s": round(toks / sum(walls), 1), "time": time.strftime("%Y-%m-%d %H:%M:%S")}), flush=True)
