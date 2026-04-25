import os
import requests
from urllib.parse import urlparse

class DownloadAgent:
    def __init__(self, cache_dir="pdf_cache"):
        self.cache_dir = cache_dir
        # Create the cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_filename(self, pdf_url):
        """Extract a safe filename from the PDF URL."""
        parsed = urlparse(pdf_url)
        path = parsed.path
        filename = os.path.basename(path)
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        return filename

    def download(self, paper):
        """
        Download the PDF for a given paper dict.
        Returns: (success: bool, local_path: str or None, message: str)
        """
        pdf_url = paper.get("pdf_url")
        title = paper.get("title", "unknown")

        if not pdf_url:
            return False, None, f"No PDF URL for {title}"

        filename = self._get_filename(pdf_url)
        local_path = os.path.join(self.cache_dir, filename)

        # Check cache
        if os.path.exists(local_path):
            return True, local_path, f"Already cached: {filename}"

        # Download
        try:
            response = requests.get(pdf_url, stream=True, timeout=30)
            response.raise_for_status()  # Raise error for bad status

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True, local_path, f"Downloaded: {filename}"
        except Exception as e:
            return False, None, f"Download failed for {title}: {str(e)}"