For SPMA's use case (symbolic event sequences, anomaly detection), best public sources:

Log anomaly datasets (closest fit)

LogHub (GitHub: logpai/loghub) — HDFS, BGL, Thunderbird, HPC, Zookeeper, Hadoop. All are structured event logs with labeled anomalies. HDFS is most studied: sequences of block operation events (WRITE, REPLICATE, DELETE) — maps directly to your symbolic sequence model.
BGL (Blue Gene/L) — supercomputer fault logs, labeled normal/anomaly per line. Events like FATAL, KERNEL, RAS — good for TRIP-like patterns.
Industrial control / power systems

HAI Security Dataset (GitHub: icsdataset/hai) — industrial control system logs from a power/water plant testbed. Closest to EDF domain. Has labeled attack/normal periods.
SWaT dataset (iTrust, Singapore) — secure water treatment plant. Structured sensor + event sequences.
BATADAL — water distribution attack dataset, sequential events.
For your specific TRIP→RESTORATION pattern

IEEE PES DataPort — power system disturbance reports. Some public. Search "protection relay event log".
ENTSO-E Transparency Platform — European grid events, but aggregated not sequence-level.
Practical recommendation: Start with LogHub/HDFS. Pre-parsed, labeled, widely benchmarked — you can compare SPMA anomaly scores against known results from other methods. BGL second for fault-log flavor closer to EDF operational data.