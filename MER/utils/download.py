import os
import requests

dir = os.path.dirname(os.path.abspath(__file__))
save_dir = dir.split("MER")[0] + "MER"

# files to download
files = {
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/"
        "opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    ),
    "res10_300x300_ssd_iter_140000.caffemodel": (
        "https://github.com/opencv/opencv_3rdparty/"
        "raw/dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    ),
}

# check if downloaded files exist
for fname in files.keys():
    if os.path.exists(os.path.join(save_dir, fname)):
        print(f"File {fname} already exists in {save_dir}. Skipping download.")
    else:
        for fname, url in files.items():
            dest = os.path.join(save_dir, fname)
            resp = requests.get(url)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)
            print(f"Downloaded {fname}")