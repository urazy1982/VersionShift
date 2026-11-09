#!/usr/bin/env python3
"""Resumable, MD5-verified CC18 ARFF prefetch for VersionShift.

HTTP transfer concurrency is separate from OpenML cache metadata operations.
This script ONLY warms the OpenML cache; the official controller freezes datasets.
Run in the modern controller environment (not Python 3.7).
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

os.environ['OPENML_SKIP_PARQUET'] = 'true'
import requests
import openml
from openml.datasets.functions import _get_cache_directory

CHUNK = 1 << 18
MIN_SEG = 2 << 20
CONNECT_TIMEOUT = 20
READ_TIMEOUT = 45
RETRIES = 4


def digest(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest().lower()


def valid(path, expected):
    return path.is_file() and path.stat().st_size > 0 and digest(path) == expected


def request_range(url, start, end):
    return requests.get(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                        allow_redirects=True, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))


def range_info(url):
    """Probe GET 0-0; tolerate no range support and use regular download."""
    try:
        with request_range(url, 0, 0) as r:
            cr = r.headers.get('Content-Range', '')
            m = re.fullmatch(r'bytes\s+0-0/(\d+)', cr)
            if r.status_code == 206 and m and int(m.group(1)) > 0:
                return r.url, int(m.group(1)), r.headers.get('ETag')
            return r.url, None, None
    except requests.RequestException:
        return url, None, None


def fetch_part(url, total, etag, start, end, dest, max_attempts):
    want = end - start + 1
    meta = dest.with_suffix(dest.suffix + '.json')
    signature = {'url': url, 'total': total, 'start': start, 'end': end, 'etag': etag}
    if dest.exists() and (not meta.exists() or json.loads(meta.read_text()) != signature):
        dest.unlink()
    meta.write_text(json.dumps(signature, sort_keys=True))
    for attempt in range(max_attempts):
        have = dest.stat().st_size if dest.exists() else 0
        if have == want:
            return
        if have > want:
            dest.unlink()
            have = 0
        try:
            with request_range(url, start + have, end) as r:
                expected_cr = f'bytes {start + have}-{end}/{total}'
                if r.status_code != 206 or r.headers.get('Content-Range', '').strip() != expected_cr:
                    raise RuntimeError(f'Invalid ranged response: HTTP {r.status_code}, Content-Range={r.headers.get("Content-Range")!r}')
                if etag and r.headers.get('ETag') and r.headers['ETag'] != etag:
                    raise RuntimeError('ETag changed during ranged download')
                received = 0
                with open(dest, 'ab') as out:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        received += len(chunk)
                        if received > want - have:
                            raise RuntimeError('Server sent more bytes than requested')
                        out.write(chunk)
                if received == 0:
                    raise RuntimeError('No response bytes received')
        except Exception as e:
            print(f'  segment {start}-{end}, attempt {attempt+1}/{max_attempts}: {str(e)[:140]}', flush=True)
            if attempt + 1 < max_attempts:
                time.sleep(min(3 * (attempt + 1), 15))
    if not dest.exists() or dest.stat().st_size != want:
        raise RuntimeError(f'Incomplete ranged segment {start}-{end}')


def download_ranged(url, size, etag, tmp, workers, attempts):
    count = max(1, min(workers, size // MIN_SEG))
    block = (size + count - 1) // count
    spans = [(start, min(size - 1, start + block - 1)) for start in range(0, size, block)]
    parts = [Path(str(tmp) + f'.part{i}') for i in range(len(spans))]
    with cf.ThreadPoolExecutor(max_workers=min(workers, len(spans))) as ex:
        futures = [ex.submit(fetch_part, url, size, etag, start, end, part, attempts)
                   for (start, end), part in zip(spans, parts)]
        for future in cf.as_completed(futures):
            future.result()
    with open(tmp, 'wb') as out:
        for part in parts:
            with open(part, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 20), b''):
                    out.write(chunk)
    if tmp.stat().st_size != size:
        raise RuntimeError('Assembled file length differs from HTTP Content-Range total')
    for part in parts:
        part.unlink()
        meta = part.with_suffix(part.suffix + '.json')
        if meta.exists():
            meta.unlink()


def download_single(url, tmp, attempts):
    # Restart single stream on failure. It is never treated as a resumable segment.
    for attempt in range(attempts):
        try:
            with requests.get(url, stream=True, allow_redirects=True,
                              timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                              headers={'Accept-Encoding': 'identity'}) as r:
                r.raise_for_status()
                with open(tmp, 'wb') as out:
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            out.write(chunk)
                advertised = r.headers.get('Content-Length')
                if advertised and tmp.stat().st_size != int(advertised):
                    raise RuntimeError('Content-Length mismatch')
            return
        except Exception as e:
            print(f'  single retry {attempt+1}/{attempts}: {str(e)[:140]}', flush=True)
            if attempt + 1 < attempts:
                time.sleep(min(4 * (attempt + 1), 20))
    raise RuntimeError('Single stream failed after bounded retries')


def verified_transfer(item, args, probed=None):
    did, name, url, expected, dest = item
    start = time.monotonic()
    temp = Path(str(dest) + '.vsdownload')
    try:
        if valid(dest, expected):
            return did, 'CACHED', 0
        # Never overwrite an unexpected cached file. Quarantine before download.
        if dest.exists():
            quarantine = Path(str(dest) + '.invalid.' + time.strftime('%Y%m%d%H%M%S'))
            dest.rename(quarantine)
            print(f'  {did}: quarantined mismatching file as {quarantine.name}', flush=True)
        final_url, size, etag = probed if probed is not None else range_info(url)
        if size is not None and size >= MIN_SEG * 2:
            download_ranged(final_url, size, etag, temp, args.connections, args.retries)
        else:
            download_single(final_url, temp, args.retries)
        if digest(temp) != expected:
            raise RuntimeError(f'MD5 mismatch expected={expected}, observed={digest(temp)}')
        # Atomic promotion only after MD5 validation. Never install partial bytes.
        os.replace(temp, dest)
        return did, 'DOWNLOADED', round(time.monotonic() - start, 2)
    except Exception as e:
        return did, 'FAILED: ' + str(e)[:200], round(time.monotonic() - start, 2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('connections', nargs='?', type=int, default=8)
    p.add_argument('--retries', type=int, default=RETRIES)
    p.add_argument('--report', default='CC18_PREFETCH_REPORT.json')
    p.add_argument('--only', type=int, nargs='*', help='Optional OpenML dataset IDs to prefetch')
    args = p.parse_args()
    if not 1 <= args.connections <= 24 or not 1 <= args.retries <= 10:
        p.error('connections must be 1..24 and retries 1..10')
    print('OpenML cache:', openml.config.get_cache_directory(), flush=True)
    print('Resolving suite 99 and metadata sequentially...', flush=True)
    suite = openml.study.get_suite(99)
    mapping = {}
    for tid in suite.tasks:
        task = openml.tasks.get_task(int(tid), download_splits=False)
        mapping[int(tid)] = int(task.dataset_id)
    ids = sorted(set(mapping.values()))
    print('CC18 tasks:', len(mapping), 'unique datasets:', len(ids), flush=True)
    if args.only is not None:
        if not set(args.only).issubset(ids):
            p.error('--only includes non-CC18 dataset ID')
        ids = [did for did in ids if did in args.only]
    items, status = [], {}
    for did in ids:
        ds = openml.datasets.get_dataset(did, download_data=False,
                                         download_qualities=False,
                                         download_features_meta_data=False)
        expected = str(ds.md5_checksum or '').strip().lower()
        if not re.fullmatch(r'[0-9a-f]{32}', expected):
            status[did] = {'status': 'FAILED: missing or invalid authoritative MD5'}
            continue
        url = ds.url
        if not url or urlparse(url).scheme != 'https':
            status[did] = {'status': 'FAILED: missing HTTPS ARFF URL'}
            continue
        folder = Path(_get_cache_directory(ds))
        dest = folder / 'dataset.arff'
        if valid(dest, expected):
            status[did] = {'status': 'CACHED', 'md5': expected, 'path': str(dest)}
        else:
            folder.mkdir(parents=True, exist_ok=True)
            items.append((did, ds.name, url, expected, dest))
    print(f'Already MD5-verified cached: {len(status)}; queued: {len(items)}', flush=True)
    # Inspect HTTP Range support before scheduling. Large files consume the entire
    # connection budget for their segments, one large file at a time. Smaller
    # non-range files download concurrently with <= connections active requests.
    ranged, normal = [], []
    for item in items:
        info = range_info(item[2])
        if info[1] is not None and info[1] >= MIN_SEG * 2:
            ranged.append((item, info))
        else:
            normal.append((item, info))
    finished = 0
    for item, info in sorted(ranged, key=lambda entry: -entry[1][1]):
        did, outcome, elapsed = verified_transfer(item, args, probed=info)
        status[did] = {'status': outcome, 'seconds': elapsed, 'md5': item[3], 'path': str(item[4])}
        finished += 1
        print(f'[{finished}/{len(items)}] {did}: {outcome} ({elapsed}s)', flush=True)
    with cf.ThreadPoolExecutor(max_workers=args.connections) as ex:
        futures = {ex.submit(verified_transfer, item, args, info): item
                   for item, info in normal}
        for future in cf.as_completed(futures):
            item = futures[future]
            did, outcome, elapsed = future.result()
            status[did] = {'status': outcome, 'seconds': elapsed, 'md5': item[3], 'path': str(item[4])}
            finished += 1
            print(f'[{finished}/{len(items)}] {did}: {outcome} ({elapsed}s)', flush=True)
    # Cache integrity checked a second time, after transfers. No OpenML cache race.
    failed = []
    for did in ids:
        record = status.get(did)
        if not record or record['status'].startswith('FAILED'):
            failed.append(did)
        elif not valid(Path(record['path']), record['md5']):
            record['status'] = 'FAILED: post-transfer MD5 validation'
            failed.append(did)
    report = {'suite_id': 99, 'task_count': len(mapping), 'unique_dataset_count': len(set(mapping.values())),
              'checked_ids': ids, 'status': {str(k): v for k, v in status.items()},
              'failed_dids': failed, 'openml_version': getattr(openml, '__version__', '?'),
              'cache': str(openml.config.get_cache_directory()),
              'note': 'Cache prefetch only. Official VersionShift controller determines eligibility and frozen folds.'}
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('Report:', args.report, 'Failures:', failed, flush=True)
    if failed:
        return 2
    return 0

if __name__ == '__main__':
    sys.exit(main())
