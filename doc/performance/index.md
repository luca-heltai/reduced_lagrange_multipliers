# Performance history

These measurements are diagnostics, not speed guarantees. GitHub-hosted CPUs
and host load vary. Compare only matching configurations and environments.
A native ratio measures overhead against deal.II at the **same commit**;
a regression compares ImmersX across **different commits**.

The table shows medians and median absolute deviations (MAD), in milliseconds.
Positive changes mean slower execution. A diagnostic regression requires an
increase greater than all of: 10%, 0.1 ms, and three times the sum of both MADs.
The comparison baseline is the first valid result in each compatible series.
Failed attempts never become timing baselines.

```{raw} html
<div id="performance-history">
  <p id="performance-status" role="status">Loading performance history…</p>
  <noscript>JavaScript is needed for filters and charts. Download the JSON history below.</noscript>
  <p><a href="data/history.json">Download complete history (JSON)</a></p>
  <div id="performance-filters"></div>
  <h2>Commit coverage</h2>
  <div id="performance-coverage"></div>
  <h2>Timing history</h2>
  <label>Chart series <select id="performance-series"></select></label>
  <div id="performance-chart"></div>
  <div id="performance-results" style="overflow-x:auto"></div>
  <h2>Attempts and validation</h2>
  <div id="performance-attempts"></div>
</div>
<script src="../_static/performance.js" defer></script>
```

The [step-80 particle penalty cases](particle-penalty) compare scalar and vector
bulk–bulk integration with a pinned native deal.II implementation, including
codimension two and independent volume meshes.

Legacy lifted and bulk–bulk trace cases have no native timing oracle recorded. Boundary
flux times different operations in ImmersX and deal.II, so its ratio is also
N/A. Cold and setup timings, numerical checks and tolerances are available in
the attempt details and downloadable raw output. Legacy CP8 trace assembly
phases stop before matrix materialization. PR #250 includes materialization and
adds explicit scope and operation counts for the nonlinear end-to-end comparison:
three ImmersX calls versus one fused MeshWorker traversal, not the same kernel.
The viewer keeps benchmark source versions and explicit scopes in separate
series. Missing scope means unspecified legacy timing, not full assembly.

See [the pipeline guide](pipeline) for environment details, commit coverage,
local execution, backfill and recovery.
