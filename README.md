# jitserve-vllm-plugin

An independent, out-of-tree scheduling plugin for **vLLM 0.31.x (V1)**, based on JITServe's Grouped Margin Goodput Maximization (GMAX). For deadline-sensitive generation, it tunes for **on-time complete responses per minute (QPM)**. It starts from live requests and learns as they finish. The caller supplies a prompt and a time budget. An output-length estimate is optional.

## Quick start

Run the following from the repository root in a working vLLM 0.31.x serving environment. The install command uses that environment's existing runtime. `MODEL` can be a local model directory or a model identifier accepted by vLLM.

```bash
python -m pip install --no-deps .
export MODEL=Qwen/Qwen3-0.6B
export VLLM_PLUGINS=jitserve_api
vllm serve "$MODEL" \
  --host 127.0.0.1 --port 8000 \
  --scheduler-cls jitserve_vllm31.scheduler.JitServeScheduler \
  --scheduling-policy priority
```

The scheduler class selects the scheduling policy implementation. The `jitserve_api` entry point enables the HTTP route. If you use other vLLM plugins, include their names in the `VLLM_PLUGINS` allowlist. For deployments that explicitly disable async scheduling, use `jitserve_vllm31.scheduler.JitServeSyncScheduler`.

```bash
curl -sS http://127.0.0.1:8000/plugins/jitserve/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Name a primary color.","timeout_s":15,"sampling_params":{"max_tokens":64,"temperature":0},"stream":false}'
```

`timeout_s` is the end-to-end scheduling budget in seconds. Configure the client's transport timeout separately. `stream: false` returns a JSON object with `text`, `finish_reason`, `output_tokens`, `ttft`, `e2e`, and `dropped`. With streaming enabled, the route emits NUL-delimited JSON frames. The deadline shorthand supports one completion per request (`n=1`). The legacy interface also accepts `request_info` and an `slo_constraint` tuple of first-token, inter-token and completion targets, in seconds.

## How the plugin learns

- **Estimate work.** Throughput requests can supply `output_len`. Otherwise the plugin can query an operator-managed QRF length predictor, or learn a p95 bound from up to 128 recently completed outputs. The initial fallback is 1024 tokens, capped by the request's `max_tokens` during ranking.
- **Select a batch.** GMAX ranks requests by estimated deadline-adjusted gain per remaining decode second, then chooses a contiguous group among candidates sorted by input length. Available admission slots bound the waiting window. The engine still enforces KV-cache and token limits.
- **Credit completed requests.** On a single local data-parallel rank, a nonempty normal-stop response earns one Q when the final HTTP body is written by the server before the deadline. Truncation, cancellation and late delivery earn zero. The boundary excludes network transit to the client. Without a local receipt channel the engine uses its terminal stop time.
- **Tune the cutoff.** Each observation window groups arrivals under one cutoff. After their outcomes settle, a contested window contributes `60 × on_time / window_seconds` to that cutoff's moving average. Warmup probes neighbouring cutoffs before continuing with epsilon-greedy exploration. Idle and uncontested windows provide no training signal.

| Setting | Default | Purpose |
| --- | --- | --- |
| `JITSERVE_GMAX` | `1` | Enable GMAX window selection |
| `JITSERVE_GMAX_ADAPTIVE` | `1` | Enable online cutoff learning |
| `JITSERVE_GMAX_CUTOFF` | `0.95` | Initial cutoff |
| `JITSERVE_GMAX_WINDOW` | `0` | Use available capacity, or set a positive cap |
| `JITSERVE_GMAX_FEEDBACK_WINDOW_S` | `30` | Observation window duration in seconds |
| `JITSERVE_MAX_WAITING_S` | `30` | Drop requests still waiting for their first schedule after this limit (`0` disables) |
| `JITSERVE_TRACE_FILE` | unset | Append cutoff, arrivals, on-time completions and QPM per finalized window |

Optional QRF lookups require `JITSERVE_QRF_HOST` and `JITSERVE_QRF_PORT`. Configure these only for a trusted predictor, as the upstream wire format uses Python pickle. Multi-stage requests can supply explicit deadlines. Historical graph matching, mid-generation QRF refinement and V0 swap-cost estimation are outside this port's scope.

## Tests

The scheduler logic and endpoint-protocol tests can run on CPU from the repository root.

```bash
python -m pip install pytest pyzmq
python -m pytest -q
```

## License and references

Distributed under the Apache License, Version 2.0. The attribution notice identifies the upstream revision and the changes specific to this plugin. [LICENSE](LICENSE) · [NOTICE](NOTICE)

- Wei Zhang, Zhiyu Wu, Yi Mu, Rui Ning, Banruo Liu, Nikhil Sarda, Myungjin Lee and Fan Lai. *JITServe: SLO-aware LLM Serving with Imprecise Request Information*. NSDI 2026, pp. 825–848. [Published paper and BibTeX](https://www.usenix.org/conference/nsdi26/presentation/zhang-wei) · [arXiv](https://arxiv.org/abs/2504.20068) · [Citation metadata](CITATION.cff)
- The original JITServe research artifact contains the reference scheduler, analyzer and evaluation workloads. [UIUC-MLSys/JITServe](https://github.com/UIUC-MLSys/JITServe)
- vLLM provides the V1 runtime and endpoint plugin interface. [vllm-project/vllm](https://github.com/vllm-project/vllm)
