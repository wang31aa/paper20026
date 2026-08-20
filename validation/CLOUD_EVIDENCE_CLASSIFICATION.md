# Cloud evidence classification

This workflow tests a frozen heterogeneous eight-domain computation on independent GitHub-hosted operating-system runners.

A passing run supports:
- clean-checkout execution;
- cross-operating-system numerical replication;
- positive within-domain parameter heterogeneity;
- agreement across direct and serialized software paths.

It does not support hardware-in-the-loop, an entity experiment, a real-time guarantee, source-controller reproduction, or physical causal recovery. The generated JSON therefore fixes `hardware_hil`, `entity_hil`, and `physical_experiment` to `false`.
