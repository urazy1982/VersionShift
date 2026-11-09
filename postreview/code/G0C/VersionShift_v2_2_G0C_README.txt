VersionShift v2.2 G0-C read-only raw-evidence audit
===================================================

1. Copy the G0-C kit ZIP to C:\VersionShift.
2. Open WSL and run:

   cd /mnt/c/VersionShift
   unzip -o VersionShift_v2_2_G0C_kit.zip
   python3 VersionShift_v2_2_G0C_evidence_audit.py \
     --zip /mnt/c/VersionShift/VersionShift_replication.zip \
     --out /mnt/c/VersionShift/V2_2_G0C

3. Outputs:
   V2_2_G0C/G0C_summary.txt
   V2_2_G0C/G0C_profiles.json
   V2_2_G0C/G0C_report_fields.csv

G0-C is an inventory/hash verification and evidence accounting stage.
It does NOT execute the imported project, run experiments, or independently
reproduce primary manuscript numbers. G0-D will perform those computations
with the now-discovered raw data schema.

No external dependencies (Python 3.8+ standard library only).
Default SHA-256 guard locks the original ZIP to the approved G0-A digest.
No extracted or original files are overwritten. Output is separate.
