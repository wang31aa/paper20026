#!/usr/bin/env python3
import json
a=json.load(open("virtual_pipe.json")); b=json.load(open("virtual_tcp.json"))
diff=max(abs(x-y) for d in a["domains"] for p in a["domains"][d]["policies"] for x,y in zip(a["domains"][d]["policies"][p],b["domains"][d]["policies"][p]))
assert diff<=1e-12
assert not a["hardware_hil"] and not b["hardware_hil"] and not a["entity_hil"] and not b["entity_hil"]
json.dump({"cross_backend_max_abs_difference":diff,"software_replication":True,"hardware_hil":False,"entity_hil":False},open("backend_comparison.json","w"),indent=2)
print("PASS cross-backend",diff)
