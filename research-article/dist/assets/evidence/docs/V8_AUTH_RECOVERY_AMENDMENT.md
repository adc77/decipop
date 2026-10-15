# V8 operational amendment: authentication recovery

Date: 2026-10-14 UTC / 2026-10-15 local. This amendment is recorded after first-seed scoring and before second-seed checkpoint selection/final tests. It changes operational execution only, with a new sealed identity; it does not rewrite either scientific freeze or claim the original recovery allowance remained unused.

## Failure and preserved evidence

The long-lived local Modal client disconnected, the ephemeral training app stopped, and its later token refresh failed with `Connection lost`. The SDK returned a cached expired short-lived JWT; the next `AttemptStart` was rejected with `AuthError: Jwt is expired`. The final planned `r2/treatment-3` reservation remains failed and fully charged against the reservation ledger, even though no final-chunk worker started. The earlier `treatment-2` call completed and committed update 756.

Fresh authenticated read-only inspection confirms the owned Volume identity, treatment state/progress at 756 with 756 trajectory entries, and the checkpoint completion marker under the unchanged scientific identity. The scheduled treatment checkpoints 900/1,134 are absent. All six then-owned corrected-study apps are stopped with zero tasks. The initial failed inspection looked for `manifest.json` instead of the checkpoint's actual `complete.json`; it made no changes or GPU calls. The corrected inspection and original controller/SDK logs are preserved.

## Exact continuation and unchanged scientific scope

Permit exactly one additional bounded L4 call to run the remaining 378 updates from the committed update-756 model, optimizer, scheduler and Python/CUDA RNG states. The worker verifies all checkpoint hashes before loading. It must reach update 1,134 and 36,288 total presentations, preserve the existing trajectory prefix and produce the originally scheduled 900/1,134 candidates. No completed training step or final-test inference is repeated.

The original parent, dataset seeds 1601/1801, all frozen data/source/config/document hashes, model architecture, loss coefficient 0.5, pair ordering, optimizer settings, exposure, selection method, calibration partitions, endpoints and scientific decision rule remain unchanged. First-seed results remain unchanged. The original scientific study identity remains authoritative; this operational permit has its own identity linked to it. No coefficient sweep, replacement seed or architecture experiment is authorized.

## Fixed total budget and call limits

Keep the original total reservation at $9. Reallocate it from $8 compute + $1 storage/transfer reserve to $8.50 compute + $0.50 storage/transfer reserve. Per-repetition compute allocations are $4 for r1 and $4.50 for r2. The extra call has the unchanged training/recovery timeout 1,050 s, startup cap 180 s, scaledown 2 s, one L4, four physical CPU cores and 32 GiB memory, retries zero. Its conservative requested-compute ceiling is $0.42613511; the full maximum including the preserved failed reservation is $8.09795067, plus the $0.50 reserve. Recheck current billing, rates and original-budget headroom before submission. Preserve failed reservations rather than refunding them in the ledger. No further authentication recovery is permitted.

Run the recovery as a fresh detached ephemeral app so client disconnection does not stop already submitted work, and explicitly stop the owned app after collecting its receipt. Continue selection/test as fresh separately bounded calls. [Modal documents](https://modal.com/docs/guide/apps) the client lifetime of ephemeral apps and the detached option. The observed cached-token behavior is from the installed SDK and the preserved log, not an assertion that the owner revoked credentials.

## Closing evidence

Record the actual resumed start/end, verified optimizer/RNG restoration, prior trajectory prefix, all scheduled candidates, the failed request and its new reservation, private checkpoint/model retention, fresh CPU reference checks, actual metering and owned-resource cleanup. Final reports must expose this deviation from the original no-further-recovery rule. Git publication remains unauthorized.
