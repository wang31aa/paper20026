#!/usr/bin/env python3
import json
from pathlib import Path
H=Path(__file__).resolve().parent;Q=json.loads((H/'MSS_OTTER_QUALIFICATION.json').read_text())
checks={'pinned_commit':len(Q['official_commit'])==40,'two_source_hashes':len(Q['official_files'])==2 and all(len(x)==64 for x in Q['official_files'].values()),'fail_closed':Q['status']=='NOT_QUALIFIED','full_difference_inventory':len(Q['v60_comparison']['not_matched'])>=10,'license_recorded':Q['license']=='MIT'}
print(json.dumps(checks,indent=2));raise SystemExit(0 if all(checks.values()) else 1)
