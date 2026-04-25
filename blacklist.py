# blacklist.py
import os

class BlacklistManager:
    def __init__(self, blacklist_file="failed_papers.txt"):
        self.blacklist_file = blacklist_file
        self._load()

    def _load(self):
        if os.path.exists(self.blacklist_file):
            with open(self.blacklist_file, 'r') as f:
                self.blacklist = set(line.strip() for line in f if line.strip())
        else:
            self.blacklist = set()

    def _save(self):
        with open(self.blacklist_file, 'w') as f:
            for arxiv_id in sorted(self.blacklist):
                f.write(arxiv_id + '\n')

    def add(self, arxiv_id: str):
        """Add a paper to the blacklist (failed download)."""
        self.blacklist.add(arxiv_id)
        self._save()

    def contains(self, arxiv_id: str) -> bool:
        return arxiv_id in self.blacklist

    def remove(self, arxiv_id: str):
        """Optional: allow removal if needed."""
        if arxiv_id in self.blacklist:
            self.blacklist.remove(arxiv_id)
            self._save()

    def clear(self):
        """Remove all entries from blacklist."""
        self.blacklist.clear()
        self._save()

    def list_all(self):
        return list(self.blacklist)