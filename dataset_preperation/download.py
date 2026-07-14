import os
import shutil
from tqdm import tqdm
from multiprocessing import get_context
import yt_dlp
import logging
from io import StringIO
import json
import argparse
from functools import partial
import time
import hashlib
import errno
import re
import subprocess
import boto3
from dotenv import load_dotenv
from download_manager import get_dataset_json_file, dataset_urls

RATE_LIMIT_COOLDOWN_SECONDS = 5 * 60
RATE_LIMIT_ERROR_PATTERNS = (
    "rate limit",
    "too many requests",
    "http error 429",
    "status code 429",
    "429:",
    "429 ",
    "requested format is not available due to rate limiting",
)
PERMANENT_VIDEO_ERROR_PATTERNS = (
    "video unavailable",
    "this video is unavailable",
    "this video is not available",
    "video is not available",
    "video not available",
    "video is private",
    "private video",
    "video has been removed",
    "video has been deleted",
    "copyright strike",
    "account associated with this video has been terminated",
    "the uploader has not made this video available",
    "sign in to confirm your age",
    "this live event will begin in",
    "this live event has ended",
    "members-only content",
)


def is_rate_limit_error(error_text):
    lowered = error_text.lower()
    return any(pattern in lowered for pattern in RATE_LIMIT_ERROR_PATTERNS)


def is_permanent_video_error(error_text):
    lowered = error_text.lower()
    if is_rate_limit_error(lowered):
        return False
    return any(pattern in lowered for pattern in PERMANENT_VIDEO_ERROR_PATTERNS)


# --------------------------------------------------------------------------------------------------
# Tigris (S3-compatible) integration
# --------------------------------------------------------------------------------------------------
#
# After every subset (folder) finishes downloading, the pipeline:
#   1. re-probes transient (e.g. rate-limit) download errors until they clear (or max rounds),
#   2. computes a sha256sum manifest for the subset,
#   3. uploads the manifest + every file in the subset folder to the Tigris bucket,
#   4. verifies that the remote objects match the local folder (names + sizes),
#   5. only then moves on to the next subset.
#
# If the disk runs out of storage at any point, the lowest-index subset already backed up to
# Tigris is removed locally to free space, disk usage is reported (df -H), and the current
# subset is restarted from the beginning (already-downloaded clips are skipped thanks to resume).

NOSPACE_MARKER = "__NOSPACE__"
TIGRIS_DEFAULT_ENDPOINT = "https://t3.storage.dev"
TIGRIS_DEFAULT_BUCKET = "genau-dataset"
TIGRIS_DEFAULT_MANIFEST_BUCKET = "genau-dataset-manifests"
SUBSET_NAME_RE = re.compile(r"^\d{6}$")
SHA256_MANIFEST_NAME = "sha256sum.txt"
FILE_LIST_NAME = "files.json"  # per-subset directory inventory uploaded to the manifest bucket
# Matches a clip file "<video_id>_<idx:03d>.<ext>" or a permanent-error file "<video_id>.error.json",
# capturing the video_id (video_ids may contain underscores, so the clip index is the LAST _NNN).
CLIP_FILE_RE = re.compile(r"^(.+)_(\d{3})\.(?:mp4|wav|json)$")
ERROR_FILE_RE = re.compile(r"^(.+)\.error\.json$")


def _env(*names, default=None):
    """Return the first non-empty environment variable among `names`, else `default`."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def is_out_of_storage_error(exc=None, error_text=""):
    """Detect an out-of-storage / ENOSPC condition from an exception or an error string."""
    if exc is not None and isinstance(exc, OSError) and exc.errno in (errno.ENOSPC,):
        return True
    lowered = (error_text or "").lower()
    return "no space left on device" in lowered or "enospc" in lowered


def get_tigris_client(endpoint_url=None, region_name=None):
    """Create an S3-compatible client pointed at Tigris.

    Endpoint, region and credentials are resolved from environment variables (loaded from a
    `.env` file via `load_dotenv()` or exported in the shell), following the Tigris quickstart.
    AWS_* names are standard; TIGRIS_* names are aliases that fill in when AWS_* is absent.
    """
    endpoint = endpoint_url or _env("AWS_ENDPOINT_URL", "TIGRIS_ENDPOINT_URL") or TIGRIS_DEFAULT_ENDPOINT
    region = region_name or _env("AWS_REGION", "TIGRIS_REGION", "AWS_DEFAULT_REGION") or "auto"
    access_key = _env("AWS_ACCESS_KEY_ID", "TIGRIS_ACCESS_KEY_ID")
    secret_key = _env("AWS_SECRET_ACCESS_KEY", "TIGRIS_SECRET_ACCESS_KEY")
    kwargs = {"endpoint_url": endpoint, "region_name": region}
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    return boto3.client("s3", **kwargs)


def compute_subset_sha256(subset_dir, manifest_name=SHA256_MANIFEST_NAME):
    """Compute a sha256sum-style manifest for every file in the subset folder.

    The manifest is written to <subset_dir>/<manifest_name> in the same format as the
    `sha256sum` coreutil: "<hexdigest>  <filename>". The manifest file itself is excluded.
    Progress is reported as bytes processed (smooth even for large media files).
    """
    # gather the files to hash and their sizes first, so we can show byte-accurate progress
    to_hash = []
    for fname in sorted(os.listdir(subset_dir)):
        fpath = os.path.join(subset_dir, fname)
        if not os.path.isfile(fpath) or fname == manifest_name:
            continue
        to_hash.append((fname, fpath, os.path.getsize(fpath)))

    entries = []
    total = sum(size for _name, _path, size in to_hash)
    label = os.path.basename(subset_dir.rstrip(os.sep)) or "subset"
    with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
              desc=f"[sha256] {label}", leave=False) as bar:
        for fname, fpath, _size in to_hash:
            digest = hashlib.sha256()
            with open(fpath, "rb") as handle:
                while True:
                    chunk = handle.read(1 << 20)
                    if not chunk:
                        break
                    digest.update(chunk)
                    bar.update(len(chunk))
            entries.append(f"{digest.hexdigest()}  {fname}")
    manifest_path = os.path.join(subset_dir, manifest_name)
    with open(manifest_path, "w") as handle:
        handle.write("\n".join(entries) + ("\n" if entries else ""))
    return manifest_path


def _upload_file_with_retry(s3, bucket, local_path, key, max_retries=3):
    last_exc = RuntimeError("upload failed: no attempts made")
    for attempt in range(max_retries):
        try:
            s3.upload_file(local_path, bucket, key)
            return
        except Exception as exc:  # noqa: BLE001 - retry any transient S3 error
            last_exc = exc
            print(f"[WARN] upload attempt {attempt + 1}/{max_retries} failed for '{key}': {exc}")
            time.sleep(2 * (attempt + 1))
    raise last_exc


def upload_subset_to_tigris(s3, data_bucket, manifest_bucket, subset_dir, subset_name,
                            manifest_name=SHA256_MANIFEST_NAME):
    """Upload a subset to Tigris, split across two buckets:

      - data files (media + json) -> <data_bucket>/<subset_name>/<filename>
      - the sha256sum manifest    -> <manifest_bucket>/<subset_name>/<manifest_name>

    Data files already present in the data bucket with a matching size are skipped (single
    prefix listing, no local log). The manifest is always (re-)uploaded last (overwriting
    whatever is there) so the manifest bucket always holds the latest checksums. Keeping the
    manifest in its own (hot) bucket means archived data objects never have to be restored for
    skip/verify reads.
    """
    local = {}
    for fname in sorted(os.listdir(subset_dir)):
        fpath = os.path.join(subset_dir, fname)
        if os.path.isfile(fpath):
            local[fname] = os.path.getsize(fpath)
    if not local:
        return

    remote = _list_remote_subset(s3, data_bucket, subset_name)
    # data files first (sorted), then the manifest last
    ordered = [f for f in sorted(local) if f != manifest_name]
    if manifest_name in local:
        ordered.append(manifest_name)

    skipped = 0
    with tqdm(ordered, desc=f"[Tigris] upload {subset_name}", leave=False) as bar:
        for fname in bar:
            fpath = os.path.join(subset_dir, fname)
            key = f"{subset_name}/{fname}"
            if fname == manifest_name:
                # manifest -> manifest bucket, always (re-)uploaded (overwrite)
                _upload_file_with_retry(s3, manifest_bucket, fpath, key)
                continue
            # data file -> data bucket; skip if already present with a matching size
            if remote.get(fname) == local[fname]:
                skipped += 1
                continue
            _upload_file_with_retry(s3, data_bucket, fpath, key)
    if skipped:
        print(f"[INFO] subset {subset_name}: {skipped}/{len(ordered)} data object(s) already present in Tigris; skipped.")
    # also store a small directory inventory in the (hot) manifest bucket for later sanity checks
    upload_subset_file_list(s3, manifest_bucket, subset_name, local)


def upload_subset_file_list(s3, manifest_bucket, subset_name, files_with_sizes,
                             file_list_name=FILE_LIST_NAME):
    """Upload a JSON inventory of the subset folder (name + size per file, plus count and total)
    to the manifest bucket. Non-critical: a failure only warns (the sha256sum manifest is the
    authoritative record used for skip/verify).
    """
    items = sorted(files_with_sizes.items())
    payload = {
        "subset": subset_name,
        "count": len(items),
        "total_size": sum(size for _name, size in items),
        "files": [{"name": name, "size": size} for name, size in items],
    }
    body = json.dumps(payload, indent=2)
    try:
        s3.put_object(Bucket=manifest_bucket, Key=f"{subset_name}/{file_list_name}",
                      Body=body.encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] could not upload file list for subset {subset_name} to manifest bucket: {exc}")


def _list_remote_subset(s3, bucket, subset_name):
    remote = {}
    prefix = f"{subset_name}/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in tqdm(paginator.paginate(Bucket=bucket, Prefix=prefix), desc=f"[Tigris] list {subset_name}", leave=False):
        for obj in page.get("Contents", []):
            fname = obj["Key"][len(prefix):]
            if fname:
                remote[fname] = int(obj.get("Size", -1))
    return remote


def _parse_sha256sum_manifest(text):
    """Parse a sha256sum-format manifest ("<hash>  <filename>" per line) into {filename: hash}."""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)  # "<hash>  <filename>"
        if len(parts) == 2:
            result[parts[1].strip()] = parts[0].strip()
    return result


def _read_local_manifest(subset_dir, manifest_name=SHA256_MANIFEST_NAME):
    local_path = os.path.join(subset_dir, manifest_name)
    if not os.path.isfile(local_path):
        return None, local_path
    with open(local_path, "r") as handle:
        return _parse_sha256sum_manifest(handle.read()), local_path


def verify_subset_in_tigris(s3, manifest_bucket, subset_dir, subset_name, manifest_name=SHA256_MANIFEST_NAME):
    """Verify a subset in Tigris by checking the *values* of the remote sha256sum manifest
    (in the manifest bucket) against the local manifest.

    The remote manifest is downloaded and its {filename: hash} entries are compared to the local
    manifest's entries. Returns (ok, details) where details has missing_in_remote, missing_in_local
    and hash_mismatch lists (plus an "error" key if the manifests could not be read).
    """
    local, local_path = _read_local_manifest(subset_dir, manifest_name)
    if local is None:
        return False, {"error": f"local manifest missing: {local_path}",
                       "missing_in_remote": [], "missing_in_local": [], "hash_mismatch": []}
    try:
        obj = s3.get_object(Bucket=manifest_bucket, Key=f"{subset_name}/{manifest_name}")
        remote = _parse_sha256sum_manifest(obj["Body"].read().decode("utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        return False, {"error": f"could not fetch remote manifest: {exc}",
                       "missing_in_remote": [], "missing_in_local": [], "hash_mismatch": []}

    missing_in_remote = sorted(k for k in local if k not in remote)
    missing_in_local = sorted(k for k in remote if k not in local)
    hash_mismatch = sorted(k for k in local if k in remote and local[k] != remote[k])
    ok = not missing_in_remote and not missing_in_local and not hash_mismatch
    return ok, {"missing_in_remote": missing_in_remote, "missing_in_local": missing_in_local,
                "hash_mismatch": hash_mismatch}


def get_remote_manifest(s3, manifest_bucket, subset_name, manifest_name=SHA256_MANIFEST_NAME):
    """Download the remote sha256sum manifest from the manifest bucket and return its
    {filename: hash} map, or None."""
    try:
        obj = s3.get_object(Bucket=manifest_bucket, Key=f"{subset_name}/{manifest_name}")
        text = obj["Body"].read().decode("utf-8", errors="replace")
        return _parse_sha256sum_manifest(text), text
    except Exception as exc:  # noqa: BLE001 - 404 / fetch error -> not uploaded
        return None, None


def is_subset_uploaded(s3, manifest_bucket, subset_name, subset_dir, manifest_name=SHA256_MANIFEST_NAME):
    """Check whether a subset is already uploaded to Tigris by comparing the *values* of the
    remote sha256sum manifest (in the manifest bucket) against the local manifest.

    - remote manifest absent -> not uploaded -> return False (re-download)
    - remote manifest present, local manifest present, values equal -> uploaded -> return True
    - remote manifest present, local manifest present, values differ -> something is wrong ->
      return False (re-process; the local/bucket manifests disagree)
    - remote manifest present, local manifest missing -> trust the remote -> return True
      (avoids re-downloading an already-backed-up subset whose local copy was pruned)
    """
    remote, _remote_text = get_remote_manifest(s3, manifest_bucket, subset_name, manifest_name)
    if remote is None:
        return False
    local, _local_path = _read_local_manifest(subset_dir, manifest_name)
    if local is None:
        return True  # can't compare; trust the remote presence
    if remote != local:
        print(f"[WARN] subset {subset_name}: remote manifest differs from local -> "
              f"something is wrong; will re-process.")
        return False
    return True


def remove_lowest_index_subset(save_dir, current_subset_name, manifest_name=SHA256_MANIFEST_NAME):
    """Free space by removing the lowest-index subset folder, but KEEP its sha256sum manifest.

    All files in the chosen subset except the manifest are deleted (media/json), leaving only the
    manifest behind. The manifest is retained so it can be compared against the remote one later
    (verifying the subset is still fully backed up in Tigris). Returns the removed subset name,
    or None when there is no other subset to remove (e.g. the current subset is 000000).
    """
    candidates = []
    for name in os.listdir(save_dir):
        full = os.path.join(save_dir, name)
        if os.path.isdir(full) and SUBSET_NAME_RE.match(name) and name != current_subset_name:
            candidates.append(name)
    if not candidates:
        return None
    lowest = min(candidates)
    lowest_dir = os.path.join(save_dir, lowest)
    for name in os.listdir(lowest_dir):
        if name == manifest_name:
            continue  # keep the manifest for later verification
        path = os.path.join(lowest_dir, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except OSError:
            pass
    return lowest


# --------------------------------------------------------------------------------------------------
# Local subset-construction check
# --------------------------------------------------------------------------------------------------
#
# A previous run (e.g. with a different --files_per_folder, or a partial/interrupted run) can leave
# clip files in the wrong subset folder. If such a "stray" file is uploaded, it lands under the
# wrong subset in Tigris and the subset's sha256sum manifest silently includes it. Before
# downloading/uploading a subset we therefore verify that every clip/error file in its folder
# actually belongs to that subset (its video_id is one of the subset's videos).

def parse_clip_video_id(fname):
    """Return the video_id embedded in a clip/error filename, or None for non-clip files."""
    m = CLIP_FILE_RE.match(fname)
    if m:
        return m.group(1)
    m = ERROR_FILE_RE.match(fname)
    if m:
        return m.group(1)
    return None


def build_video_to_subset_map(all_entries, files_per_folder):
    """Map each video_id -> its correct 6-digit subset name (based on global video index)."""
    vid_to_subset = {}
    for video_idx, (video_id, _info) in all_entries:
        vid_to_subset[video_id] = f"{video_idx // files_per_folder:06d}"
    return vid_to_subset


def find_stray_files(subset_dir, subset_entries):
    """Return clip/error files in the subset folder whose video_id does NOT belong to this subset.

    These are videos misplaced from other subsets by previous runs. Non-clip files (the manifest,
    .DS_Store, etc.) are ignored.
    """
    vids = {entry[1][0] for entry in subset_entries}
    strays = []
    if not os.path.isdir(subset_dir):
        return strays
    for fname in os.listdir(subset_dir):
        vid = parse_clip_video_id(fname)
        if vid is not None and vid not in vids:
            strays.append(fname)
    return sorted(strays)


def reorganize_local_subsets(save_dir, vid_to_subset):
    """One-time pass: move every misplaced clip/error file into its correct subset folder.

    Use BEFORE uploading any subsets to Tigris (e.g. on the first Tigris run after an older run
    with a different --files_per_folder). Moving a file into a subset that is already uploaded to
    Tigris would desync the local copy from the remote manifest, so don't run this after uploads.
    Returns (moved, skipped) where skipped covers out-of-range videos and name conflicts.
    """
    moved = 0
    skipped = 0
    for name in os.listdir(save_dir):
        sub = os.path.join(save_dir, name)
        if not os.path.isdir(sub) or not SUBSET_NAME_RE.match(name):
            continue
        for fname in os.listdir(sub):
            vid = parse_clip_video_id(fname)
            if vid is None:
                continue  # manifest / junk -> leave it
            correct = vid_to_subset.get(vid)
            if correct is None:
                skipped += 1  # video not in this run's dataset range -> can't place it
                continue
            if correct == name:
                continue  # already in the right folder
            dst_dir = os.path.join(save_dir, correct)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, fname)
            if os.path.exists(dst):
                skipped += 1  # conflict; don't overwrite
                continue
            shutil.move(os.path.join(sub, fname), dst)
            moved += 1
    return moved, skipped



def show_disk_usage():
    print("[INFO] Current disk usage (df -H):")
    try:
        subprocess.run(["df", "-H"], check=False)
    except FileNotFoundError:
        usage = shutil.disk_usage(os.getcwd())
        print(f"[INFO] (df not available) total={usage.total} used={usage.used} free={usage.free}")


def clean_temp_dir(temp_working_dir):
    """Remove leftover worker scratch directories after a pool termination."""
    if not temp_working_dir or not os.path.isdir(temp_working_dir):
        return
    for name in os.listdir(temp_working_dir):
        path = os.path.join(temp_working_dir, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except OSError:
            pass


def write_json_file(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle)


def build_video_error_path(outpath, video_id):
    return os.path.join(outpath, f"{video_id}.error.json")


def register_rate_limit_cooldown(rate_state, rate_lock, cooldown_seconds=RATE_LIMIT_COOLDOWN_SECONDS):
    if rate_state is None or rate_lock is None:
        return

    with rate_lock:
        cooldown_until_ts = time.monotonic() + cooldown_seconds
        rate_state["cooldown_until_ts"] = max(rate_state.get("cooldown_until_ts", 0.0), cooldown_until_ts)
        rate_state["next_allowed_ts"] = max(rate_state.get("next_allowed_ts", 0.0), rate_state["cooldown_until_ts"])


def wait_for_global_download_slot(rate_state, rate_lock, max_videos_per_hour):
    if rate_state is None or rate_lock is None or max_videos_per_hour <= 0:
        return

    min_interval_seconds = 3600.0 / float(max_videos_per_hour)
    while True:
        with rate_lock:
            now = time.monotonic()
            next_allowed_ts = rate_state.get("next_allowed_ts", 0.0)
            cooldown_until_ts = rate_state.get("cooldown_until_ts", 0.0)
            wait_time = max(next_allowed_ts, cooldown_until_ts) - now
            if wait_time <= 0:
                rate_state["next_allowed_ts"] = now + min_interval_seconds
                return
        time.sleep(min(wait_time, 0.25))

def download_yt_video(entry,
                    save_dir,
                    temp_working_dir,
                    rate_state=None,
                    rate_lock=None,
                    max_videos_per_hour=1700.0,
                    yt_cookie_path=None,
                    audio_only=False,
                    proxy=None,
                    audio_sampling_rate=44100,
                    resume=True,
                    files_per_folder=5000,
                    sleep_interval=10.0,
                    sleep_interval_subtitles=5.0,
                    sleep_interval_requests=0.75,
                    max_sleep_interval=20.0):
    video_idx = entry[0]
    video_id, intervals = entry[1][0], entry[1][1]['intervals']
    subfolder_idx = f'{video_idx // files_per_folder:06}'
    outpath = os.path.join(save_dir, subfolder_idx)
    try:
        os.makedirs(outpath, exist_ok=True)
    except OSError as _e:
        if is_out_of_storage_error(_e):
            print(f"[FATAL] out of storage while creating subset folder {outpath}: {_e}")
            return f"{NOSPACE_MARKER}: creating {outpath}: {_e}"
        raise
    video_error_path = build_video_error_path(outpath, video_id)

    if resume and os.path.isfile(video_error_path):
        print(f"[INFO] skipping permanently failed video {video_id} because {video_error_path} exists")
        return None

    for file_idx, video_info in enumerate(intervals):
        start = video_info['start']
        to = video_info['end']
        autocap_caption = video_info.get('text', None)

        clip_json_path = os.path.join(outpath, f'{video_id}_{file_idx:03d}.json')

        if resume and os.path.isfile(clip_json_path):
            continue
        else:
            ytdl_logger = logging.getLogger()
            log_stream = StringIO()
            logging.basicConfig(stream=log_stream, level=logging.INFO)

            out_file_ext = 'wav' if audio_only else 'mp4'
            format = 'bestaudio/best' if audio_only else 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio'
            ydl_opts = {
                "logger": ytdl_logger,
                'outtmpl': f"{temp_working_dir}/id_{video_id}_{file_idx:03d}/audio.%(ext)s",
                'format': format,
                'quiet': True,
                'ignoreerrors': False,
                # 'writesubtitles': True,  # Attempt to download subtitles (transcripts)
                # 'writeautomaticsub': True,  # Attempt to download automatic subtitles (auto-generated transcripts)
                'force_generic_extractor': True,
                'postprocessor_args': ['-ar', str(audio_sampling_rate)],
                'external_downloader':'ffmpeg',
                'download_ranges': yt_dlp.utils.download_range_func([], [[start, to]]),
                'force-keyframe-at-cuts': True,
                'external_downloader_args':['-loglevel', 'quiet'],
                "remote_components": ["ejs:github"],
                "cookiesfrombrowser": ("firefox",),
            }
            if sleep_interval > 0:
                ydl_opts["sleep_interval"] = sleep_interval
            if sleep_interval_subtitles > 0:
                ydl_opts["sleep_interval_subtitles"] = sleep_interval_subtitles
            if sleep_interval_requests > 0:
                ydl_opts["sleep_interval_requests"] = sleep_interval_requests
            if max_sleep_interval > 0:
                ydl_opts["max_sleep_interval"] = max_sleep_interval
            if yt_cookie_path is not None:
                ydl_opts['cookiefile'] = f'{temp_working_dir}/id_{video_id}_{file_idx:03d}/cookies.txt'
            if audio_only:
                ydl_opts['postprocessors'] = [{'key': 'FFmpegExtractAudio',
                                               'preferredcodec': 'wav'}]
            else:
                ydl_opts['postprocessors'] = [{'key': 'FFmpegVideoConvertor',
                                                'preferedformat': 'mp4',  # Ensure the output is MP4
                                                }]
            if proxy is not None:
                ydl_opts['proxy'] = f'socks5://127.0.0.1:{proxy}/'

            url = f'https://www.youtube.com/watch?v={video_id}'
            temp_clip_dir = f'{temp_working_dir}/id_{video_id}_{file_idx:03d}'
            os.makedirs(temp_clip_dir, exist_ok=True)
            if yt_cookie_path is not None:
                shutil.copy(yt_cookie_path, f'{temp_clip_dir}/cookies.txt')
            try:
                wait_for_global_download_slot(rate_state=rate_state,
                                              rate_lock=rate_lock,
                                              max_videos_per_hour=max_videos_per_hour)
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    file_exist = os.path.isfile(os.path.join(outpath, f'{video_id}_{file_idx:03d}.{out_file_ext}'))
                    info=ydl.extract_info(url, download=not file_exist)
                    filename = f'{video_id}_{file_idx:03d}.{out_file_ext}'
                    jsonname = f'{video_id}_{file_idx:03d}.json'
                    if not file_exist:
                        shutil.move(os.path.join(temp_clip_dir, f'audio.{out_file_ext}'), os.path.join(outpath, filename))
                    else:
                        pass
                    file_meta = {'id':f'{video_id}','path': os.path.join(outpath, filename),'title': info['title'], 'url':url, 'start': start, 'end': to}

                    if autocap_caption is not None:
                        file_meta['autocap_caption'] = autocap_caption

                    # meta data
                    file_meta['resolution'] = info.get('resolution')
                    file_meta['fps'] = info.get('fps')
                    file_meta['aspect_ratio'] = info.get('aspect_ratio')
                    file_meta['audio_channels'] = info.get('audio_channels')

                    file_meta['description'] = info.get('description')
                    file_meta['uploader'] = info.get('uploader')
                    file_meta['upload_date'] = info.get('upload_date')
                    file_meta['duration'] = info.get('duration')
                    file_meta['view_count'] = info.get('view_count')
                    file_meta['like_count'] = info.get('like_count')
                    file_meta['channel_follower_count'] = info.get('channel_follower_count')
                    file_meta['dislike_count'] = info.get('dislike_count')
                    file_meta['channel_id'] = info.get('channel_id')
                    file_meta['channel_url'] = info.get('channel_url')
                    file_meta['channel_name'] = info.get('uploader')

                    print("[INFO] save meta data for", os.path.join(outpath, jsonname))
                    write_json_file(os.path.join(outpath, jsonname), file_meta)
                shutil.rmtree(temp_clip_dir, ignore_errors=True)
            except Exception as e:
                shutil.rmtree(temp_clip_dir, ignore_errors=True)
                error_text = f'{url} - ytdl : {log_stream.getvalue()}, system : {str(e)}'
                if is_out_of_storage_error(e, error_text):
                    print(f"[FATAL] out of storage while downloading {clip_json_path}: {e}")
                    return f"{NOSPACE_MARKER}: {error_text}"
                if is_rate_limit_error(error_text):
                    register_rate_limit_cooldown(rate_state=rate_state,
                                                 rate_lock=rate_lock,
                                                 cooldown_seconds=RATE_LIMIT_COOLDOWN_SECONDS)
                    print(f"[ERROR] rate limited while downloading {clip_json_path}: {e}")
                    return error_text

                if is_permanent_video_error(error_text):
                    error_meta = {
                        "id": video_id,
                        "url": url,
                        "status": "error",
                        "retryable": False,
                        "failed_interval_index": file_idx,
                        "start": start,
                        "end": to,
                        "error": str(e),
                        "log": log_stream.getvalue(),
                    }
                    write_json_file(video_error_path, error_meta)
                    print(f"[ERROR] marked video {video_id} as permanently failed at {video_error_path}: {e}")
                else:
                    print(f"[ERROR] downloading {clip_json_path}:", e)
                return error_text
    return None

def update_interval_dict(dict_1, dict_2):
    """
    combine two dictionaries, and merge intervals list if it is replicated
    """
    for k, v in dict_2.items():
        if k in dict_1:
            dict_2[k]['intervals'] += dict_1[k]['intervals']

    dict_1.update(dict_2)

def read_video_segments_info(local_input_video_segments,
                             start_idx=0,
                             end_idx=int(1e9),
                             min_audio_len=0.0,
                            clap_threshold=0.0):
    all_video_segments = {}
    total_number_of_clips = 0
    with open(local_input_video_segments, 'r') as f:
        last_idx = 0
        for idx, json_str in enumerate(tqdm(f, desc="parsing json input")):
            if idx > start_idx:
                try:
                    json_str = json_str.strip()
                    if json_str.endswith('\n'):
                        json_str = json_str[:-1]
                    if json_str.endswith(','):
                        json_str = json_str[:-1]

                    data = json.loads(json_str)
                    video_ids = list(data.keys())
                    if len(video_ids) == 0:
                        continue
                    video_id = video_ids[0]

                    intervals = data[video_id].get("intervals", [])
                    len_intervals_filtered = [clip for clip in intervals if float(clip['end']) - float(clip['start'])>= min_audio_len]
                    clap_len_intervals_filtered = [clip for clip in len_intervals_filtered if  clip.get("CLAP_SIM", -9999) is not None and clip.get("CLAP_SIM", -9999) >= clap_threshold]
                    total_number_of_clips += len(clap_len_intervals_filtered)
                    video_data = {}
                    video_data[video_id] = {}
                    video_data[video_id]['intervals'] = clap_len_intervals_filtered
                    update_interval_dict(all_video_segments, video_data)
                except Exception as e:
                    print("[ERROR] Couldn't parse json string:", json_str)
                    continue
                last_idx += 1

            if last_idx >= end_idx:
                break

    print(f"Found {total_number_of_clips} audio clips.")
    return all_video_segments

# Module-level worker glue so the spawn-based pool can map over entries while returning
# which entry each result belongs to (needed to retry only the failed entries).
_WORKER_DOWNLOAD_FN = None


def _init_pool_worker(download_fn):
    global _WORKER_DOWNLOAD_FN
    _WORKER_DOWNLOAD_FN = download_fn


def _download_entry(entry):
    """Run download_yt_video for one entry and tag the result with its video index."""
    if _WORKER_DOWNLOAD_FN is None:  # not initialized (should not happen with the pool initializer)
        raise RuntimeError("pool worker download function was not initialized")
    result = _WORKER_DOWNLOAD_FN(entry)
    return (entry[0], result)


def _write_logs(logs, path):
    try:
        with open(path, "w") as handle:
            handle.write("\n".join(str(log) for log in logs if log is not None))
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] could not write logs to {path}: {exc}")


def _download_subset_via_pool(pool, subset_entries, pbar, counted_idxs):
    """Download a subset's entries through the pool.

    Returns (results, nospace) where results is a list of (video_idx, result) and nospace is
    True if any worker reported an out-of-storage condition. On NOSPACE we stop consuming
    early; the caller is responsible for terminating/recreating the pool.

    The progress bar is advanced at most ONCE per entry (the first time it is processed), so
    re-probing a persistently-failing entry does not inflate the bar. `counted_idxs` tracks
    which entry indices have already been counted and persists across probe rounds and
    out-of-storage restarts within the same subset.
    """
    results = []
    nospace = False
    for video_idx, result in pool.imap_unordered(_download_entry, subset_entries):
        if video_idx not in counted_idxs:
            counted_idxs.add(video_idx)
            pbar.update()
        results.append((video_idx, result))
        if isinstance(result, str) and result.startswith(NOSPACE_MARKER):
            nospace = True
            break
    return results, nospace


def download_audioset_split(json_file,
                            save_dir,
                            temp_working_dir,
                            yt_cookie_path,
                            audio_only=False,
                            proxy_port=None,
                            audio_sampling_rate=44100,
                            start_idx=0,
                            end_idx=int(1e9),
                            num_processes=os.cpu_count(),
                            resume=True,
                            files_per_folder=5000,
                            clap_threshold=0.4,
                            min_audio_len=4,
                            max_videos_per_hour=1700.0,
                            sleep_interval=10.0,
                            sleep_interval_subtitles=5.0,
                            sleep_interval_requests=0.75,
                            max_sleep_interval=20.0,
                            upload_to_tigris=True,
                            tigris_bucket=TIGRIS_DEFAULT_BUCKET,
                            tigris_manifest_bucket=TIGRIS_DEFAULT_MANIFEST_BUCKET,
                            tigris_endpoint=None,
                            max_retry_rounds=3,
                            reorganize_local=False,
                            subset_log_path="download_logs.txt",
                            ):

    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Reading video segments information from {json_file}...")
    print(f"[INFO] Temp working directory for downloads is set to {temp_working_dir}")

    # Resolve Tigris settings from the environment (.env file / exported env vars) when not
    # provided explicitly. Real env vars win over the .env file (see load_dotenv in __main__).
    if tigris_bucket is None:
        tigris_bucket = _env("TIGRIS_BUCKET", default=TIGRIS_DEFAULT_BUCKET)
    if tigris_manifest_bucket is None:
        tigris_manifest_bucket = _env("TIGRIS_MANIFEST_BUCKET", default=TIGRIS_DEFAULT_MANIFEST_BUCKET)
    if tigris_endpoint is None:
        tigris_endpoint = _env("AWS_ENDPOINT_URL", "TIGRIS_ENDPOINT_URL")

    num_processes = num_processes or os.cpu_count() or 1

    all_video_segments = read_video_segments_info(json_file,
                                                  start_idx=start_idx,
                                                  end_idx=end_idx,
                                                  min_audio_len=min_audio_len,
                                                  clap_threshold=clap_threshold)

    # Group the (video_idx, (video_id, info)) entries into subsets keyed by folder index.
    all_entries = list(enumerate(all_video_segments.items(), start=start_idx))
    subsets = {}
    for entry in all_entries:
        subset_idx = entry[0] // files_per_folder
        subsets.setdefault(subset_idx, []).append(entry)
    sorted_subset_idxs = sorted(subsets.keys())
    if not sorted_subset_idxs:
        print("[INFO] No video segments to download.")
        return
    print(f"[INFO] {len(all_entries)} videos grouped into {len(sorted_subset_idxs)} subsets "
          f"(files_per_folder={files_per_folder}): "
          f"{sorted_subset_idxs[0]:06d}..{sorted_subset_idxs[-1]:06d}")

    # Map every video_id to the subset folder it belongs in, used to detect/relocate stray files.
    vid_to_subset = build_video_to_subset_map(all_entries, files_per_folder)
    if reorganize_local:
        print("[INFO] Reorganizing local subset folders (moving misplaced clips to their "
              f"correct subsets). Use this BEFORE uploading subsets to Tigris.")
        moved, skipped = reorganize_local_subsets(save_dir, vid_to_subset)
        print(f"[INFO] Reorganize done: {moved} file(s) moved, {skipped} skipped "
              f"(out-of-range videos / name conflicts).")

    s3 = get_tigris_client(tigris_endpoint) if upload_to_tigris else None
    if upload_to_tigris:
        print(f"[INFO] Tigris uploads enabled: data bucket='{tigris_bucket}' "
              f"manifest bucket='{tigris_manifest_bucket}' "
              f"endpoint={tigris_endpoint or TIGRIS_DEFAULT_ENDPOINT}")
        print("[INFO] Manifests are read from the (hot) manifest bucket; data objects are only "
              f"listed/uploaded (works on archived data buckets).")
    else:
        print("[INFO] Tigris uploads disabled; subsets will only be kept locally.")

    ctx = get_context("spawn")
    with ctx.Manager() as manager:
        rate_state = manager.dict()
        rate_state["next_allowed_ts"] = 0.0
        rate_state["cooldown_until_ts"] = 0.0
        rate_lock = manager.Lock()

        download_audio_split = partial(download_yt_video,
                                       save_dir=save_dir,
                                       temp_working_dir=temp_working_dir,
                                       rate_state=rate_state,
                                       rate_lock=rate_lock,
                                       max_videos_per_hour=max_videos_per_hour,
                                       yt_cookie_path=yt_cookie_path,
                                       audio_only=audio_only,
                                       proxy=proxy_port,
                                       audio_sampling_rate=audio_sampling_rate,
                                       resume=resume,
                                       files_per_folder=files_per_folder,
                                       sleep_interval=sleep_interval,
                                       sleep_interval_subtitles=sleep_interval_subtitles,
                                       sleep_interval_requests=sleep_interval_requests,
                                       max_sleep_interval=max_sleep_interval)

        all_logs = []
        pool = ctx.Pool(num_processes * 2, initializer=_init_pool_worker,
                        initargs=(download_audio_split,))
        try:
            with tqdm(total=len(all_entries), desc="download") as pbar:
                for subset_idx in sorted_subset_idxs:
                    subset_name = f"{subset_idx:06d}"
                    subset_dir = os.path.join(save_dir, subset_name)
                    subset_entries = subsets[subset_idx]
                    entry_by_idx = {e[0]: e for e in subset_entries}
                    pbar.set_postfix_str(f"subset={subset_name} n={len(subset_entries)}")
                    print(f"\n[INFO] === Subset {subset_name}: {len(subset_entries)} videos ===")

                    # ------------------------------------------------------------------
                    # 0. Resume: skip subsets already uploaded to Tigris. We ask the bucket
                    #    directly (no local log): GET the remote sha256sum manifest and compare its
                    #    values to the local manifest. Equal -> already uploaded -> skip. Not equal ->
                    #    something is wrong -> re-process. Absent -> not uploaded -> download.
                    # ------------------------------------------------------------------
                    if upload_to_tigris and is_subset_uploaded(s3, tigris_manifest_bucket, subset_name, subset_dir):
                        print(f"[INFO] subset {subset_name} already uploaded (remote manifest "
                              f"matches local); skipping download & upload.")
                        pbar.update(len(subset_entries))
                        continue

                    # ------------------------------------------------------------------
                    # 0.5. Local-construction check: refuse to download/upload a subset folder
                    #      that contains clip files belonging to OTHER subsets (misplaced by a
                    #      previous run, e.g. with a different --files_per_folder). Run once with
                    #      --reorganize_local to fix, or remove the stray files manually.
                    # ------------------------------------------------------------------
                    strays = find_stray_files(subset_dir, subset_entries)
                    if strays:
                        preview = ", ".join(strays[:5]) + ("..." if len(strays) > 5 else "")
                        print(f"[ERROR] subset {subset_name} is not correctly constructed: "
                              f"{len(strays)} file(s) belong to other subsets (e.g. {preview}). "
                              f"Refusing to upload a contaminated subset. "
                              f"Run with --reorganize_local to move them to the right subsets, "
                              f"or remove them manually, then rerun.")
                        _write_logs(all_logs, subset_log_path)
                        return

                    # ------------------------------------------------------------------
                    # 1-2. Download subset, probing transient (rate-limit) errors again.
                    #     On out-of-storage: prune the lowest-index subset and restart.
                    # ------------------------------------------------------------------
                    counted_idxs = set()  # bump the bar at most once per entry (no overshoot on re-probes / restarts)
                    while True:
                        round_entries = subset_entries
                        transient_remaining = False
                        for attempt in range(max(1, max_retry_rounds)):
                            results, nospace = _download_subset_via_pool(pool, round_entries, pbar, counted_idxs)
                            if nospace:
                                break

                            # record non-NOSPACE errors for the log file
                            for _vi, result in results:
                                if result is not None and not result.startswith(NOSPACE_MARKER):
                                    all_logs.append(result)

                            failed_idxs = [vi for vi, result in results
                                          if result is not None and not is_permanent_video_error(result)]
                            if not failed_idxs:
                                transient_remaining = False
                                break
                            transient_remaining = True
                            print(f"[INFO] subset {subset_name}: {len(failed_idxs)} transient error(s) "
                                  f"after probe round {attempt + 1}; probing again...")
                            round_entries = [entry_by_idx[vi] for vi in failed_idxs]
                            # give rate limits some time to clear before re-probing
                            time.sleep(min(RATE_LIMIT_COOLDOWN_SECONDS, 30))

                        if nospace:
                            print(f"[FATAL] out of storage while downloading subset {subset_name}.")
                            removed = remove_lowest_index_subset(save_dir, subset_name)
                            show_disk_usage()
                            if removed is None:
                                print(f"[FATAL] No lower-index subset available to remove for "
                                      f"{subset_name}. Stopping to avoid data loss.")
                                _write_logs(all_logs, subset_log_path)
                                return
                            print(f"[INFO] Removed lowest-index subset {removed} to free space. "
                                  f"Restarting subset {subset_name} from the start.")
                            clean_temp_dir(temp_working_dir)
                            pool.terminate()
                            pool.join()
                            pool = ctx.Pool(num_processes * 2, initializer=_init_pool_worker,
                                            initargs=(download_audio_split,))
                            continue  # restart this subset from the beginning (resume skips done clips)

                        if transient_remaining:
                            print(f"[ERROR] subset {subset_name} still has transient errors after "
                                  f"{max_retry_rounds} probe round(s). Stopping.")
                            _write_logs(all_logs, subset_log_path)
                            return
                        break  # subset download complete, proceed to sha256 + upload

                    # ------------------------------------------------------------------
                    # 3. Compute the sha256sum manifest for this subset.
                    # ------------------------------------------------------------------
                    print(f"[INFO] Computing sha256sum for subset {subset_name}...")
                    try:
                        compute_subset_sha256(subset_dir)
                    except OSError as exc:
                        if not is_out_of_storage_error(exc):
                            raise
                        print(f"[FATAL] out of storage while writing sha256sum for {subset_name}.")
                        removed = remove_lowest_index_subset(save_dir, subset_name)
                        show_disk_usage()
                        if removed is None:
                            print(f"[FATAL] No lower-index subset to remove. Stopping.")
                            _write_logs(all_logs, subset_log_path)
                            return
                        print(f"[INFO] Removed lowest-index subset {removed} to free space.")
                        clean_temp_dir(temp_working_dir)
                        compute_subset_sha256(subset_dir)

                    # ------------------------------------------------------------------
                    # 4-5. Upload the subset (+ manifest) to Tigris and verify it matches.
                    # ------------------------------------------------------------------
                    if upload_to_tigris:
                        print(f"[INFO] Uploading subset {subset_name}: data -> '{tigris_bucket}', "
                              f"manifest -> '{tigris_manifest_bucket}'...")
                        upload_subset_to_tigris(s3, tigris_bucket, tigris_manifest_bucket,
                                                subset_dir, subset_name)

                        ok, details = verify_subset_in_tigris(
                            s3, tigris_manifest_bucket, subset_dir, subset_name)
                        if not ok:
                            print(f"[WARN] subset {subset_name} manifest verify mismatch -> "
                                  f"{details}; re-uploading manifest and re-checking.")
                            mpath = os.path.join(subset_dir, SHA256_MANIFEST_NAME)
                            if os.path.isfile(mpath):
                                _upload_file_with_retry(s3, tigris_manifest_bucket, mpath,
                                                        f"{subset_name}/{SHA256_MANIFEST_NAME}")
                            ok, details = verify_subset_in_tigris(
                                s3, tigris_manifest_bucket, subset_dir, subset_name)

                        if not ok:
                            print(f"[ERROR] subset {subset_name} failed manifest verification in Tigris. "
                                  f"Stopping. details={details}")
                            _write_logs(all_logs, subset_log_path)
                            return
                        n_files = sum(1 for f in os.listdir(subset_dir)
                                      if os.path.isfile(os.path.join(subset_dir, f)))
                        print(f"[INFO] subset {subset_name} verified in Tigris (manifest values match; "
                              f"{n_files} local files).")
                    else:
                        print(f"[INFO] Tigris upload skipped for subset {subset_name}.")

                    print(f"[INFO] subset {subset_name} complete.")
        finally:
            pool.close()
            pool.join()

    _write_logs(all_logs, subset_log_path)

if __name__ == "__main__":
    import tempfile
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name",
                        type=str,
                        required=True,
                        help=f"Provided the dataset names. Available datasets are {dataset_urls.keys()}")

    parser.add_argument("--clap_threshold",
                        type=float,
                        required=False,
                        default=0.4,
                        help=f"Provided the clap similarity threshold to filter the dataset, default: 0.4")

    parser.add_argument("--min_audio_len",
                        type=float,
                        required=False,
                        default=4,
                        help=f"Provided the minimum audio clip length to filter the dataset, default: 4s")

    parser.add_argument("--input_file",
                        type=str,
                        default=None,
                        required=False,
                        help="Provided the path to the json object that contains the dataset information. You may leave it empty to attempt to download the required files from the web")

    parser.add_argument("--save_dir",
                        type=str,
                        required=False,
                        default='data/datasets/autocap/videos',
                        help="where to save the downloaded files")

    parser.add_argument("--audio_only",
                        required=False,
                        action='store_true',
                        help="Enable to only save the wav files and discard the vidoes")

    parser.add_argument("--cookie_path",
                        type=str,
                        required=False,
                        default=None,
                        help="Path to your Youtube cookies files")

    parser.add_argument("--sampling_rate",
                        type=int,
                        default=44100,
                        help="Audio sampling rate, default is set to 44.1KHz")

    parser.add_argument("--proxy",
                        type=str,
                        default=None,
                        help="provde a proxy port to bypass youtube blocking your IP")

    parser.add_argument("--files_per_folder",
                        type=int,
                        default=50000,
                        help="How many files to store per folder")

    parser.add_argument('--start_idx', '-s',
                        type=int, default=0,
                        help="start index of the json objects in the provided files")

    parser.add_argument('--end_idx', '-e', type=int, default=int(1e9),
                        help="start index of the json objects in the provided files")

    parser.add_argument('--redownload', action='store_true',
                        help="redownload already downloaded files")

    parser.add_argument('--num_processes', type=int, default=os.cpu_count(),
                        help="number of processes to use for downloading, default is set to the number of CPU cores")

    parser.add_argument("--max_videos_per_hour",
                        type=float,
                        default=1700.0,
                        help="Global cap on started video downloads across all workers")

    parser.add_argument("--sleep_interval",
                        type=float,
                        default=0.0,
                        help="Global and yt-dlp sleep interval lower bound")

    parser.add_argument("--sleep_interval_subtitles",
                        type=float,
                        default=0.0,
                        help="yt-dlp subtitle request sleep interval")

    parser.add_argument("--sleep_interval_requests",
                        type=float,
                        default=0.0,
                        help="Global and yt-dlp minimum gap between requests")

    parser.add_argument("--max_sleep_interval",
                        type=float,
                        default=0.0,
                        help="Global and yt-dlp sleep interval upper bound")

    parser.add_argument("--tigris_bucket",
                        type=str,
                        default=None,
                        help="Tigris bucket for the dataset data objects (default: env TIGRIS_BUCKET or 'genau-dataset'). Set this bucket's default tier to GLACIER/Archive for cheap storage.")

    parser.add_argument("--tigris_manifest_bucket",
                        type=str,
                        default=None,
                        help="Tigris bucket for the per-subset sha256sum.txt manifests (default: env TIGRIS_MANIFEST_BUCKET or 'genau-dataset-manifests'). Keep this bucket in the Standard tier so manifests are instantly readable.")

    parser.add_argument("--tigris_endpoint",
                        type=str,
                        default=None,
                        help="Tigris S3 endpoint URL (default: env AWS_ENDPOINT_URL / TIGRIS_ENDPOINT_URL or https://t3.storage.dev)")

    parser.add_argument("--skip_tigris_upload",
                        action='store_true',
                        help="Do not upload subsets to Tigris; keep them only on local disk")

    parser.add_argument("--max_retry_rounds",
                        type=int,
                        default=3,
                        help="Max probe rounds to retry transient (e.g. rate-limit) download errors per subset")

    parser.add_argument("--reorganize_local",
                        action='store_true',
                        help="One-time pass BEFORE downloading: move clip files sitting in the wrong subset folder (from a previous run with a different --files_per_folder) into their correct subset folder. Use BEFORE uploading subsets to Tigris.")

    args = parser.parse_args()

    # Load Tigris credentials/config from a .env file in the current directory
    # (real env vars already set in the shell still take precedence).
    load_dotenv()

    with tempfile.TemporaryDirectory() as temp_dir:
        if args.input_file is None or not os.path.exists(args.input_file):
            args.input_file = get_dataset_json_file(args.dataset_name, args.input_file, download=True)

        download_audioset_split(json_file=args.input_file,
                                save_dir=args.save_dir,
                                temp_working_dir=temp_dir,
                                audio_only=args.audio_only,
                                audio_sampling_rate=args.sampling_rate,
                                yt_cookie_path=args.cookie_path,
                                proxy_port=args.proxy,
                                start_idx=args.start_idx,
                                end_idx=args.end_idx,
                                clap_threshold=args.clap_threshold,
                                min_audio_len=args.min_audio_len,
                                max_videos_per_hour=args.max_videos_per_hour,
                                sleep_interval=args.sleep_interval,
                                sleep_interval_subtitles=args.sleep_interval_subtitles,
                                sleep_interval_requests=args.sleep_interval_requests,
                                max_sleep_interval=args.max_sleep_interval,
                                resume=not args.redownload,
                                files_per_folder=args.files_per_folder,
                                num_processes=args.num_processes,
                                upload_to_tigris=not args.skip_tigris_upload,
                                tigris_bucket=args.tigris_bucket,
                                tigris_manifest_bucket=args.tigris_manifest_bucket,
                                tigris_endpoint=args.tigris_endpoint,
                                max_retry_rounds=args.max_retry_rounds,
                                reorganize_local=args.reorganize_local
        )
