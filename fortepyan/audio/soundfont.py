import urllib
from pathlib import Path

import appdirs

from fortepyan import config as C


def download_if_needed() -> Path:
    """
    Ensures the SoundFont file is downloaded in the user's data directory.

    The function checks for 'soundfont.sf2' in the user's data directory, downloading it if absent.
    The download URL is sourced from the application's configuration.

    Returns:
        Path: The path to the SoundFont file.
    """
    app_name = "fortepyan"
    app_author = "me"

    data_dir = appdirs.user_data_dir(app_name, app_author)
    data_dir = Path(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / "soundfont.sf2"

    if not file_path.exists():
        print("Downloading SoundFont")
        urllib.request.urlretrieve(C.SOUNDFONT_URL, file_path)

    return file_path
