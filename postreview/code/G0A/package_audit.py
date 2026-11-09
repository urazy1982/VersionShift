#!/usr/bin/env python3
"""Read-only ZIP intake audit for VersionShift v2.2.

Usage:
  python VersionShift_v2_2_package_audit.py --zip /path/to/VersionShift_replication.zip --out /path/to/audit

The script never extracts or edits source files. It validates member CRCs,
calculates SHA-256 hashes, and creates an inventory for subsequent content audits.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile


def checksum(path, block=1048576):
    h=hashlib.sha256()
    with open(path,"rb") as fp:
        for b in iter(lambda: fp.read(block), b""):
            h.update(b)
    return h.hexdigest()


def entry_flags(name, info):
    norm = name.replace('\\', '/')
    parts = PurePosixPath(norm).parts
    flags=[]
    if norm.startswith('/') or (len(norm) > 1 and norm[1] == ':'):
        flags.append('absolute_path')
    if '..' in parts:
        flags.append('parent_traversal')
    if info.flag_bits & 1:
        flags.append('encrypted')
    if info.file_size > (1 << 30):
        flags.append('large_member_over_1GiB')
    if info.compress_size and info.file_size / info.compress_size > 1000:
        flags.append('high_compression_ratio')
    return flags


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--zip', required=True, dest='zip_path')
    a.add_argument('--out', required=True)
    args=a.parse_args()
    source=Path(args.zip_path).expanduser().resolve()
    if not source.is_file():
        sys.exit('ZIP not found: ' + str(source))
    if not zipfile.is_zipfile(str(source)):
        sys.exit('Not a readable ZIP file: ' + str(source))
    output=Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    members=[]
    names=collections.Counter()
    suffixes=collections.Counter()
    flags=collections.Counter()
    totals=collections.Counter()
    with zipfile.ZipFile(str(source),'r') as z:
        for info in z.infolist():
            names[info.filename] += 1
            if info.is_dir():
                continue
            ef=entry_flags(info.filename, info)
            h=hashlib.sha256()
            read_bytes=0
            error=''
            try:
                with z.open(info) as stream:
                    while True:
                        chunk=stream.read(1048576)
                        if not chunk:
                            break
                        read_bytes += len(chunk)
                        h.update(chunk)
            except Exception as exc:
                error=type(exc).__name__ + ': ' + str(exc)
                ef.append('read_or_crc_error')
            for tag in ef: flags[tag] += 1
            suffix=Path(info.filename).suffix.lower() or '[no extension]'
            suffixes[suffix] += 1
            totals['uncompressed_bytes'] += info.file_size
            totals['compressed_bytes'] += info.compress_size
            members.append({
                'path':info.filename,
                'uncompressed_bytes':info.file_size,
                'compressed_bytes':info.compress_size,
                'crc32_hex':'%08x' % info.CRC,
                'sha256':h.hexdigest() if not error else '',
                'read_bytes':read_bytes,
                'suffix':suffix,
                'flags':';'.join(ef),
                'error':error,
            })

    duplicate_names=sorted([name for name,count in names.items() if count > 1])
    manifest={
        'source_basename':source.name,
        'archive_size_bytes':source.stat().st_size,
        'archive_sha256':checksum(source),
        'member_file_count':len(members),
        'member_directory_count':sum(1 for n in names if n.endswith('/')),
        'total_uncompressed_bytes':totals['uncompressed_bytes'],
        'total_compressed_bytes':totals['compressed_bytes'],
        'extension_counts':dict(sorted(suffixes.items())),
        'flag_counts':dict(sorted(flags.items())),
        'duplicate_member_names':duplicate_names,
        'members':members,
        'note':'Archive integrity and inventory only. Statistical outputs, provenance, and experiment correctness are NOT yet verified.',
    }
    with (output/'package_manifest.json').open('w',encoding='utf-8') as fp:
        json.dump(manifest,fp,indent=2,ensure_ascii=False)
    with (output/'file_inventory.csv').open('w',encoding='utf-8',newline='') as fp:
        cols=['path','uncompressed_bytes','compressed_bytes','crc32_hex','sha256','read_bytes','suffix','flags','error']
        writer=csv.DictWriter(fp,fieldnames=cols)
        writer.writeheader()
        writer.writerows(members)
    lines=[
        'VersionShift ZIP intake summary',
        'Source: '+str(source),
        'Archive SHA-256: '+manifest['archive_sha256'],
        'Files: '+str(len(members)),
        'Uncompressed bytes: '+str(totals['uncompressed_bytes']),
        'Duplicate member names: '+str(len(duplicate_names)),
        'Flagged members: '+str(sum(1 for r in members if r['flags'])),
        'Member read/CRC errors: '+str(sum(1 for r in members if r['error'])),
        'IMPORTANT: This is not an analysis replication or scientific certification.',
    ]
    (output/'audit_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    if duplicate_names or flags['read_or_crc_error'] or flags['absolute_path'] or flags['parent_traversal']:
        sys.exit(2)

if __name__=='__main__':
    main()
