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
    os.makedirs(outpath, exist_ok=True)
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
                            ):

    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Reading video segments information from {json_file}...")
    print(f"[INFO] Temp working directory for downloads is set to {temp_working_dir}")

    all_video_segments = read_video_segments_info(json_file,
                                                  start_idx=start_idx,
                                                  end_idx=end_idx,
                                                  min_audio_len=min_audio_len,
                                                  clap_threshold=clap_threshold)

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

        logs = []
        p = ctx.Pool(num_processes*2)

        # download_audio_split = partial(save_metadata, split=split) # save_metadata
        with tqdm(total=len(all_video_segments),leave=False) as pbar:
            for log in p.imap_unordered(download_audio_split, enumerate(all_video_segments.items(), start=start_idx)):
                logs.append(log)
                pbar.update()
        p.close()
        p.join()
    logs = [l for l in logs if l is not None]
    open(f'download_logs.txt','w').write('\n'.join(logs))

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

    args = parser.parse_args()

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
                                num_processes=args.num_processes
        )
