# CPU Thread Binding Exploration Memory

## Scope
This memory records the CPU worker thread-affinity exploration around `Qwen3.5` serving on CPU, with emphasis on why `serve` behaved worse than `bench latency` on the SMT-enabled AArch64 server.

Target branch context during this work: `v0.18.1-gdn-cpu`.

## Key Lessons

### 1. The main worker-thread regression started at `084d98b33`
That change widened the worker caller thread back to the full bind mask after the initial OpenMP pinning pass in `csrc/cpu/utils.cpp`.

Effect:
- before that commit, the caller thread effectively stayed on one core after the first OMP region
- after that commit, it was explicitly rebound to the whole worker CPU group

For server workloads this was harmful because the worker main/control thread participates in execution and spawns later helper/runtime threads.

### 2. `serve` vs `bench latency` was not a pure platform difference
The initial suspicion was x86-vs-AArch64 behavior, but the more important distinction was worker-thread role and later runtime thread creation.

Important observations:
- x86 tolerated the broader main-thread mask better
- AArch64 was more sensitive to it
- the root issue was still the widened worker caller thread and inherited affinity of later threads, not only platform ISA

### 3. The old behavior works better: do not rebind the worker caller after the OMP loop
Best result from these experiments:
- keep the initial one-to-one OMP worker pinning
- do not explicitly rebind the worker caller thread afterward

This means `init_cpu_threads_env()` should stop after the OMP pinning loop instead of restoring either:
- the full CPU mask
- or an explicit first-core mask

The caller is already participating in the initial OpenMP region, so an additional post-loop rebinding step is unnecessary and can be harmful.

### 4. There are at least three competing CPU thread pools in the real workload
The worker process is not just one OMP team.

Observed categories during serving:
- vLLM native CPU ops / OpenMP threads
- PyTorch operator/runtime threads
- torch compiled / C++ compiled kernel threads

This explains why per-core behavior can remain messy even after the worker OMP team is pinned correctly.

### 5. Dedicated execution-thread experiments were useful for diagnosis but not adopted here
A full dedicated-thread model was explored in a separate debug branch. It helped confirm:
- `worker_control` can be separated from compute
- hidden thread pools appear mainly after `load_model()` and after message queue initialization
- many threads inherit whatever mask the execution thread has at creation time

However, the final conclusion for `v0.18.1-gdn-cpu` was not to merge that exploration. The stable branch should keep the simpler model and only avoid the harmful post-loop rebinding.

### 6. `thread_siblings_list` is more reliable than `lscpu CORE` for SMT grouping on AArch64
On the SMT AArch64 machine, grouping by `lscpu -e=CPU,CORE,NODE` alone was not reliable enough for sibling-aware planning.

For any future SMT-aware work, prefer:
- `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`

over only the `CORE` field from `lscpu`.

### 7. Message-queue / distributed-runtime threads appear after worker init completes
Debug snapshots showed a meaningful stage split:
- `post_load_model`: backend/runtime pools from model load
- `worker_init_complete`: additional unmanaged threads after `_init_message_queues()`

That means some thread spread is not from model kernels directly, but from communicator / distributed runtime initialization.

### 8. Runtime unification matters on AArch64
Local testing and user experiments suggested that unifying runtimes around:
- `libomp.so`
- torch compile with LLVM

reduced competition between the different CPU thread pools.

That is an environment/runtime result, not a source-tree change in this checkpoint, but it is important operational knowledge.

## Validation Notes
Validation for the final stable change on `v0.18.1-gdn-cpu` was limited to:
- rebuilding CPU extensions with the required local env
- verifying `import vllm._custom_ops`

Commands used:
```bash
source $HOME/virtualenv/venv_vllm/bin/activate
OMP_NUM_THREADS=4 VLLM_CPU_OMP_THREADS_BIND=0-3 VLLM_TARGET_DEVICE=cpu \
  taskset -c 0-3 python3 setup.py build_ext --inplace

PYTHONPATH=/home/weihe/workspace/vllm-oss VLLM_TARGET_DEVICE=cpu \
OMP_NUM_THREADS=4 VLLM_CPU_OMP_THREADS_BIND=0-3 \
  taskset -c 0-3 python3 - <<'PY'
import vllm._custom_ops
print("IMPORT_OK")
PY
```

## Commit Anchors
- `084d98b33` `[CPU][AArch64] fix binding, FP8 routing, and compile threads`
  - introduced the harmful full-mask restore of the worker caller thread
- `a28a3d489` `[CPU][Debug] checkpoint worker thread exploration`
  - preserved the larger dedicated-thread / debug exploration on branch `debug/cpu-worker-thread-exploration-20260413`
- `070b1f840` `[CPU][Affinity] pin worker main thread to first bound core`
  - intermediate rollback step before removing post-loop rebinding entirely
