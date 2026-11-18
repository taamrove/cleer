# PLAN

## Done
- [x] Plugin: feedback suppressor + expander + RNNoise noise remover + dereverb (C++)
- [x] Noise remover on a worker thread (no glitching at 64-sample buffers)
- [x] NR **Live** mode: zero-latency min-phase filter steered by RNNoise (default); **Quality** = through-the-model

## Next
- [ ] Live mode is C++-only for now (no RNNoise stage in `reference/` or the Swift app) - port only if wanted
- [ ] Listen-test Live vs Quality on more vocals; retune onset bypass / gain smoothing if needed
- [ ] Low-latency time-domain filterbank denoise in the Swift app (sub-10 ms full chain)
