# Heterogeneous collective capability: code-only archive

This repository contains source programs, tests, plotting programs and the
machine-readable configurations required to inspect the computational methods.
It deliberately excludes the manuscript, mathematical proof documents,
internal audits, raw and derived data, frozen numerical results, rendered
figures and private author material.

The archive is therefore a source-code disclosure, not a complete result
reproduction package. The full evidence package can be supplied to editors and
reviewers under the applicable data and licensing conditions. Third-party data
must be obtained from the repositories cited by the article.

## Integrity check

```bash
python tools/validate_code_only_release.py
python -m compileall -q .
```

The integrity check rejects manuscript, proof, audit, result and data paths,
checks every manifest hash and confirms that no file exceeds GitHub's limit.

No physical experiment, HIL execution, public DOI or independent reproduction
is claimed by this code archive.

