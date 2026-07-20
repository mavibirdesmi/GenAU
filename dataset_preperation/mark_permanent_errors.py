import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from download_manager import get_dataset_json_file, dataset_urls

DEFAULT_END_IDX = int(1e9)


def resolve_input_row_bounds(start_idx=None, end_idx=None, files_per_folder=5000,
                             subset=None, subset_start=None, subset_end=None):
    if files_per_folder <= 0:
        raise ValueError("files_per_folder must be greater than 0")

    legacy_args_used = start_idx is not None or end_idx is not None
    single_subset_used = subset is not None
    subset_range_used = subset_start is not None or subset_end is not None
    range_modes_used = sum([legacy_args_used, single_subset_used, subset_range_used])
    if range_modes_used > 1:
        raise ValueError("Choose exactly one range selector: --start_idx with --end_idx, "
                         "--subset, or --subset_start with --subset_end")

    if legacy_args_used:
        if start_idx is None or end_idx is None:
            raise ValueError("--start_idx and --end_idx must be provided together")
        if start_idx < 0:
            raise ValueError("start_idx must be greater than or equal to 0")
        if end_idx <= start_idx:
            raise ValueError("end_idx must be greater than start_idx")
        return start_idx, end_idx

    if single_subset_used:
        if subset < 0:
            raise ValueError("subset must be greater than or equal to 0")
        return subset * files_per_folder, (subset + 1) * files_per_folder

    if subset_range_used:
        if subset_start is None or subset_end is None:
            raise ValueError("--subset_start and --subset_end must be provided together")
        if subset_start < 0:
            raise ValueError("subset_start must be greater than or equal to 0")
        if subset_end <= subset_start:
            raise ValueError("subset_end must be greater than subset_start")
        return subset_start * files_per_folder, subset_end * files_per_folder

    return 0, DEFAULT_END_IDX


def build_video_index_map(local_input_video_segments, start_idx=0, end_idx=DEFAULT_END_IDX,
                          min_audio_len=0.0, clap_threshold=0.0):
    all_video_segments = {}
    with open(local_input_video_segments, 'r') as handle:
        for idx, json_str in enumerate(handle):
            if idx < start_idx:
                continue
            if idx >= end_idx:
                break

            try:
                json_str = json_str.strip()
                if json_str.endswith(','):
                    json_str = json_str[:-1]

                data = json.loads(json_str)
                video_ids = list(data.keys())
                if not video_ids:
                    continue
                video_id = video_ids[0]

                intervals = data[video_id].get("intervals", [])
                filtered = [
                    clip for clip in intervals
                    if float(clip['end']) - float(clip['start']) >= min_audio_len
                ]
                filtered = [
                    clip for clip in filtered
                    if clip.get("CLAP_SIM", -9999) is not None
                    and clip.get("CLAP_SIM", -9999) >= clap_threshold
                ]
                if not filtered:
                    continue

                if video_id not in all_video_segments:
                    all_video_segments[video_id] = {
                        "raw_idx": idx,
                        "interval_count": 0,
                    }
                all_video_segments[video_id]["interval_count"] += len(filtered)
            except Exception:
                print(f"[WARN] Could not parse input row {idx}; skipping")
                continue
    return all_video_segments


def read_video_ids(video_ids, video_ids_file):
    ordered = []
    seen = set()

    for video_id in video_ids or []:
        video_id = video_id.strip()
        if video_id and video_id not in seen:
            seen.add(video_id)
            ordered.append(video_id)

    if video_ids_file is not None:
        with open(video_ids_file, 'r') as handle:
            for line in handle:
                video_id = line.strip()
                if not video_id or video_id.startswith('#'):
                    continue
                if video_id not in seen:
                    seen.add(video_id)
                    ordered.append(video_id)

    return ordered


def build_video_error_path(save_dir, raw_idx, files_per_folder, video_id):
    subset_name = f"{raw_idx // files_per_folder:06d}"
    subset_dir = os.path.join(save_dir, subset_name)
    return subset_name, subset_dir, os.path.join(subset_dir, f"{video_id}.error.json")


def write_json_file(path, payload):
    with open(path, 'w') as handle:
        json.dump(payload, handle)


def build_error_payload(video_id, raw_idx, subset_name, input_file, reason, interval_count):
    return {
        "id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "status": "error",
        "retryable": False,
        "source": "manual_review",
        "manually_marked_permanent": True,
        "raw_idx": raw_idx,
        "subset": subset_name,
        "interval_count": interval_count,
        "input_file": os.path.abspath(input_file),
        "error": reason,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create permanent-error JSON files for manually reviewed YouTube video IDs."
    )
    parser.add_argument("--dataset_name",
                        type=str,
                        default=None,
                        help=f"Dataset name. Available datasets are {dataset_urls.keys()}")
    parser.add_argument("--input_file",
                        type=str,
                        default=None,
                        help="Path to the dataset JSONL file. If omitted, it is resolved via --dataset_name.")
    parser.add_argument("--save_dir",
                        type=str,
                        required=True,
                        help="Dataset root containing or intended to contain subset folders like 000000, 000001, ...")
    parser.add_argument("--video_id",
                        action="append",
                        default=None,
                        help="YouTube video ID to mark as permanently failed. Repeat this flag for multiple IDs.")
    parser.add_argument("--video_ids_file",
                        type=str,
                        default=None,
                        help="Text file with one YouTube video ID per line. Lines starting with # are ignored.")
    parser.add_argument("--reason",
                        type=str,
                        default="Manually reviewed and marked as permanently unavailable after yt-dlp could not classify it.",
                        help="Reason written into each generated .error.json file.")
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="Overwrite existing .error.json files instead of skipping them.")
    parser.add_argument("--clap_threshold",
                        type=float,
                        default=0.4,
                        help="CLAP similarity threshold used when locating each video in the filtered dataset.")
    parser.add_argument("--min_audio_len",
                        type=float,
                        default=4,
                        help="Minimum clip duration used when locating each video in the filtered dataset.")
    parser.add_argument("--files_per_folder",
                        type=int,
                        default=50000,
                        help="How many raw input rows map to each 6-digit subset folder.")
    parser.add_argument('--start_idx', '-s',
                        type=int, default=None,
                        help="Legacy raw JSONL row start index, inclusive. Must be used with --end_idx.")
    parser.add_argument('--end_idx', '-e',
                        type=int, default=None,
                        help="Legacy raw JSONL row end index, exclusive. Must be used with --start_idx.")
    parser.add_argument("--subset",
                        type=int,
                        default=None,
                        help="Only resolve video IDs within one 6-digit subset by raw row index.")
    parser.add_argument("--subset_start",
                        type=int,
                        default=None,
                        help="First 6-digit subset index to consider, inclusive.")
    parser.add_argument("--subset_end",
                        type=int,
                        default=None,
                        help="Last 6-digit subset index to consider, exclusive.")

    args = parser.parse_args()

    if args.input_file is None:
        if args.dataset_name is None:
            parser.error("Provide either --input_file or --dataset_name")
        args.input_file = get_dataset_json_file(args.dataset_name, args.input_file, download=True)

    video_ids = read_video_ids(args.video_id, args.video_ids_file)
    if not video_ids:
        parser.error("Provide at least one --video_id or --video_ids_file")

    try:
        start_idx, end_idx = resolve_input_row_bounds(
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            files_per_folder=args.files_per_folder,
            subset=args.subset,
            subset_start=args.subset_start,
            subset_end=args.subset_end,
        )
    except ValueError as exc:
        parser.error(str(exc))

    save_dir = os.path.abspath(args.save_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    video_map = build_video_index_map(
        args.input_file,
        start_idx=start_idx,
        end_idx=end_idx,
        min_audio_len=args.min_audio_len,
        clap_threshold=args.clap_threshold,
    )

    created = 0
    skipped = 0
    missing = []

    for video_id in video_ids:
        info = video_map.get(video_id)
        if info is None:
            missing.append(video_id)
            continue

        raw_idx = info["raw_idx"]
        interval_count = info["interval_count"]
        subset_name, subset_dir, error_path = build_video_error_path(
            save_dir, raw_idx, args.files_per_folder, video_id
        )
        Path(subset_dir).mkdir(parents=True, exist_ok=True)

        if os.path.exists(error_path) and not args.overwrite:
            print(f"[INFO] skipping existing permanent error file: {error_path}")
            skipped += 1
            continue

        payload = build_error_payload(
            video_id=video_id,
            raw_idx=raw_idx,
            subset_name=subset_name,
            input_file=args.input_file,
            reason=args.reason,
            interval_count=interval_count,
        )
        write_json_file(error_path, payload)
        print(f"[INFO] wrote manual permanent error file for {video_id} -> {error_path}")
        created += 1

    print(f"[INFO] created {created} permanent error file(s); skipped {skipped} existing file(s)")
    if missing:
        print("[WARN] the following video id(s) were not found in the selected filtered dataset range:")
        for video_id in missing:
            print(video_id)


if __name__ == '__main__':
    main()
