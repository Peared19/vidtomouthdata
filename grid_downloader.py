import os
import requests
import tarfile
import zipfile
from tqdm import tqdm
import warnings

# Warningok elnyomása
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Mappák létrehozása (alapértelmezett struktúra)
base_dirs = [
    "gridcorpus/raw/audio",
    "gridcorpus/raw/video",
    "gridcorpus/raw/align",
    "gridcorpus/audio",
    "gridcorpus/video",
    "gridcorpus/align"
]

for d in base_dirs:
    os.makedirs(d, exist_ok=True)

# Felhasználótól bekérés
start_speaker = int(input("Enter the starting speaker number: "))
end_speaker = int(input("Enter the ending speaker number: "))
extract_files = input("Do you want to extract files after downloading? (y/n): ").lower()

def download_file(url, dest_path):
    """Letöltés progress barral, biztosítva a célmappa létét"""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        response = requests.get(url, stream=True, verify=False)
        if response.status_code == 200:
            total = int(response.headers.get('content-length', 0))
            with open(dest_path, 'wb') as file, tqdm(
                desc=os.path.basename(dest_path),
                total=total,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            return True
        else:
            print(f"⚠️  Failed: {url} ({response.status_code})")
            return False
    except Exception as e:
        print(f"⚠️  Error downloading {url}: {e}")
        return False

for i in range(start_speaker, end_speaker + 1):
    speaker_id = f"s{i}"
    print(f"\n------------------------- Downloading speaker {speaker_id} -------------------------")

    audio_url = f"https://spandh.dcs.shef.ac.uk/gridcorpus/{speaker_id}/audio/{speaker_id}.tar"
    video_url = f"https://spandh.dcs.shef.ac.uk/gridcorpus/{speaker_id}/video/{speaker_id}.mpg_vcd.zip"
    align_url = f"https://spandh.dcs.shef.ac.uk/gridcorpus/{speaker_id}/align/{speaker_id}.tar"

    audio_path = f"gridcorpus/raw/audio/{speaker_id}.tar"
    video_path = f"gridcorpus/raw/video/{speaker_id}.zip"
    align_path = f"gridcorpus/raw/align/{speaker_id}.tar"

    # Letöltések
    audio_ok = download_file(audio_url, audio_path)
    video_ok = download_file(video_url, video_path)
    align_ok = download_file(align_url, align_path)

    # Kicsomagolás
    if extract_files == "y":
        if audio_ok and os.path.exists(audio_path):
            print(f"→ Extracting audio for {speaker_id}")
            with tarfile.open(audio_path, "r") as tar_ref:
                tar_ref.extractall(f"gridcorpus/audio/{speaker_id}")

        if video_ok and os.path.exists(video_path):
            print(f"→ Extracting video for {speaker_id}")
            with zipfile.ZipFile(video_path, "r") as zip_ref:
                zip_ref.extractall(f"gridcorpus/video/{speaker_id}")

        if align_ok and os.path.exists(align_path):
            print(f"→ Extracting transcriptions for {speaker_id}")
            with tarfile.open(align_path, "r") as tar_ref:
                tar_ref.extractall(f"gridcorpus/align/{speaker_id}")

print("\n✅ All downloads, transcriptions, and extractions completed successfully!")
