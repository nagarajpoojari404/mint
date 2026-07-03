import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path


ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

url = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
filename = "eval_bundle.zip"

Path("data").mkdir(exist_ok=True)
Path("dump").mkdir(exist_ok=True)

opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
urllib.request.install_opener(opener)
urllib.request.urlretrieve(url, filename)  # noqa: S310

if Path(filename).exists():
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall("data")
    shutil.move(filename, "dump/eval_bundle.zip")
