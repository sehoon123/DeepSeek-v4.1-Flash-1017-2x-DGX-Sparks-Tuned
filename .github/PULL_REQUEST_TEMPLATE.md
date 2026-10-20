## Description and Motivation

<!--

    Please write a description of what this PR is changing, removing or adding, and why.
    Consider including before/after comparisons.

    For this recipe, a good description usually covers:
      * which setting in scripts/config.sh, script behavior, pin (TF_REF, INPUTS_REF, KIT_REF) or patch changes
      * whether the change affects measured numbers (prefill/decode throughput, TTFT, concurrent requests, the
        memory left on the head) and in which direction
      * for a patch in zig/patches: whether it keeps the same bits, and that drafted replies still equal plain ones

-->

## Related Issues

<!--

    Add the list of issues related to this PR from the [issue tracker](https://github.com/sehoon123/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks-Tuned/issues).
    Indicate which of these issues are resolved or fixed by this PR, like #XXXX, where XXXX is the issue number.

-->

---

## Testing

<!--

    Tell us how you verified this change. For this recipe that usually means:

      * `bash -n start.sh stop.sh scripts/*.sh` (syntax check)
      * `python3 -I -m py_compile tools/*.py scripts/zig_flags.py`
      * `shellcheck start.sh stop.sh scripts/*.sh` if available
      * `DRY_RUN=1 ./start.sh` (the commands it prints)
      * `scripts/prepare.sh` (a patch or pin change rebuilds the image on the head and copies it to the worker; every
        patch must apply with `git apply` in the engine checkout)
      * an actual launch on two Sparks, plus `tools/validate.py`, `tools/toolcheck.py` and `tools/needle.py` against the
        running server, and sparkDash (https://github.com/MiaAI-Lab/sparkDash) for speed
      * if behavior changed, the measured numbers with the new settings, stating which configuration they came from

    If you changed a patch, confirm drafted replies still equal plain ones: the same request with DRAFTS=0 and a fixed
    "seed" (or temperature 0) must give the same reply. Say whether the patch keeps the same bits as before.

-->

---

## Checklist:

<!--

    Thanks for contributing!

    Before you file this pull request, please follow the items on this checklist and
    put an x in each of the boxes, like this: [x].

-->

- [ ] I have read the README and `scripts/config.sh` and kept my changes consistent with them.
- [ ] My pull request has a sound title and description (not something vague like `Update README.md`).
- [ ] My change is reproducible and verified (script syntax check, `DRY_RUN=1 ./start.sh`, `scripts/prepare.sh`, a launch, or a re-measurement).
- [ ] A patch change keeps drafted replies equal to plain ones, and I said how I checked it (and whether its bits change).
- [ ] I updated the README and/or `scripts/config.sh` if a setting, default, or measured number changed.
- [ ] If my change affects memory, I checked how much the head has left at the default 4 x 1,048,576-token setting.
- [ ] Defaults in `scripts/config.sh` still work out of the box; a new setting has a sane fallback like the existing ones.
