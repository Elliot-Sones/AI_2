import os
from tensorflow.python.summary.summary_iterator import summary_iterator

# Find the event file
log_dir = "/root/AI_2/AI_2/results/ppo_utmist/tb/PPO_8"
event_file = [f for f in os.listdir(log_dir) if "tfevents" in f][0]
path = os.path.join(log_dir, event_file)

print(f"Inspecting {path}...")

tags = set()
try:
    for e in summary_iterator(path):
        for v in e.summary.value:
            tags.add(v.tag)
except Exception as e:
    print(f"Error reading events: {e}")

print("Found tags:")
for t in sorted(tags):
    print(t)
