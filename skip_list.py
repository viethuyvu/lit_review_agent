# skip_list.py
import os

class SkipListManager:
    def __init__(self, skip_file="skipped_papers.txt"):
        self.skip_file = skip_file
        self._load()

    def _load(self):
        if os.path.exists(self.skip_file):
            with open(self.skip_file, 'r') as f:
                self.skipped = set(line.strip() for line in f if line.strip())
        else:
            self.skipped = set()

    def _save(self):
        with open(self.skip_file, 'w') as f:
            for arxiv_id in sorted(self.skipped):
                f.write(arxiv_id + '\n')

    def add(self, arxiv_id: str):
        self.skipped.add(arxiv_id)
        self._save()

    def contains(self, arxiv_id: str) -> bool:
        return arxiv_id in self.skipped

    def remove(self, arxiv_id: str):
        if arxiv_id in self.skipped:
            self.skipped.remove(arxiv_id)
            self._save()

    def clear(self):
        self.skipped.clear()
        self._save()

    def list_all(self):
        return list(self.skipped)