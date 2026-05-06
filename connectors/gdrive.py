import io
import os
import logging
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

logger = logging.getLogger(__name__)

# Read-only access to Drive is all we need
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SERVICE_ACCOUNT_FILE = "service_account.json"


def get_drive_service():
    """Authenticates and returns the Google Drive API service client."""
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise FileNotFoundError(
            f"Missing {SERVICE_ACCOUNT_FILE}. "
            "Download it from Google Cloud Console and place it in the project root."
        )
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)


def list_drive_files(folder_id: str) -> list:
    """
    Fetches metadata (id, name, modifiedTime, mimeType) for ALL files in a
    specific folder, handling Google Drive API pagination automatically.
    """
    service = get_drive_service()
    query = f"'{folder_id}' in parents and trashed = false"

    all_files = []
    page_token = None

    while True:
        kwargs = dict(
            q=query,
            fields="nextPageToken, files(id, name, modifiedTime, mimeType)",
            pageSize=100,
        )
        if page_token:
            kwargs["pageToken"] = page_token

        response = service.files().list(**kwargs).execute()
        all_files.extend(response.get("files", []))

        page_token = response.get("nextPageToken")
        if not page_token:
            break  # No more pages — we have everything

    logger.info("Listed %d files from Drive folder %s", len(all_files), folder_id)
    return all_files


def download_file_to_disk(
    file_id: str, file_name: str, mime_type: str, download_dir: str = "temp_downloads"
) -> str:
    """Downloads a file from Google Drive and saves it locally for parsing.
    Natively exports Google Docs to plain text."""
    service = get_drive_service()
    
    # Google Docs cannot be downloaded directly, they must be exported
    if mime_type == "application/vnd.google-apps.document":
        request = service.files().export_media(fileId=file_id, mimeType="text/plain")
        if not file_name.endswith(".txt"):
            file_name += ".txt"
    else:
        request = service.files().get_media(fileId=file_id)

    os.makedirs(download_dir, exist_ok=True)
    file_path = os.path.join(download_dir, file_name)

    with open(file_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            logger.debug("Downloading %s ... %.0f%%", file_name, (status.progress() * 100))

    logger.info("Downloaded '%s' to %s", file_name, file_path)
    return file_path