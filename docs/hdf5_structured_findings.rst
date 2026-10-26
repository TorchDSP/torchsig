Structured HDF5 Benchmark Findings
==================================

Decision target
---------------

The initial project gate is a minimum 1.5x shuffled end-to-end throughput
improvement over the matching online pipeline. A recommendation requires three
repetitions with no more than 5% throughput standard deviation. Application
owners should set a stricter target before running when their storage budget or
training schedule requires it.

Evidence available so far
-------------------------

A 100,000-sample synthetic Stage-2 run used 4,096-element complex IQ, batch
size 64, fp32, the deterministic fallback transform/model, and a full-dataset
layout. The confirmation run used four workers, uncompressed one-sample chunks,
100 measured steps, 10 warmup steps, and three repetitions.

* Materialized shuffled throughput was 19,355.9 samples/s versus 5,555.6
  samples/s online: a 3.48x speedup that passes the 1.5x gate.
* Materialized sequential throughput was 16,641.5 samples/s versus 5,837.1
  samples/s online: a 2.85x speedup.
* The shuffled path reached 62.5% of the 30,951.8 samples/s device-only ceiling.
* The structured copy required 3.12 GiB and amortized materialization after
  approximately 4.63 epochs.
* Repetition stability passed the 5% standard-deviation criterion.

An earlier configuration matrix found uncompressed one-sample chunks to be the
best tested choice. LZF and 32-sample chunks frequently reduced shuffled
throughput, sometimes below the online baseline. Four workers improved the
winning shuffled configuration in that run. Physical split separation was not
measured in the confirmation run, so no performance claim is made for it.

Conclusion and limits
---------------------

The current evidence supports uncompressed one-sample chunks as a conservative
starting point for this deterministic fallback workload. It does not establish
a universal default: production transform/model factories, the IQ scalar,
spectrogram, and fixed-shape detection workloads, full versus separate layouts,
and representative target hardware must still be measured with the enhanced
benchmark. Large generated datasets and raw result files remain untracked;
reviewable JSON/CSV artifacts should be attached to the MR or retained by the
project's benchmark system.
