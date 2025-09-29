import gdown, zipfile, os

# Google Drive file ID
file_id = "19SpbUAiUltDRlhlN_ziOBhZjMf2EU0fO"
url = f"https://drive.google.com/uc?id={file_id}"

# Download zip
output = "dataset.zip"
gdown.download(url, output, quiet=False)

# Extract zip
with zipfile.ZipFile(output, "r") as zip_ref:
    zip_ref.extractall("dataset")

print("✅ Extracted folders:", os.listdir("dataset"))


