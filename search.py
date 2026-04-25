# search.py
import arxiv
import time
from filter import FilterAgent

class SearchAgent:
    def __init__(self, blacklist, skip_list, max_papers=200):
        """
        blacklist: BlacklistManager instance
        skip_list: SkipListManager instance
        max_papers: maximum number of papers to fetch from arXiv per query
        """
        self.blacklist = blacklist
        self.skip_list = skip_list
        self.max_papers = max_papers

    def fetch_relevant_papers(self, topic: str, target_count: int, threshold: float = 6.0) -> list:
        """
        Fetch up to max_papers papers, filter out blacklisted/skipped,
        use Claude to score relevance, keep those >= threshold,
        and add low-relevance papers to skip list.
        Returns list of relevant papers (each dict with 'relevance_score').
        """
        from filter import FilterAgent
        filter_agent = FilterAgent()
        collected = []
        search = arxiv.Search(query=topic, max_results=self.max_papers, sort_by=arxiv.SortCriterion.Relevance)
        client = arxiv.Client()
        batch_papers = []
        try:
            for result in client.results(search):
                paper = {
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "summary": result.summary,
                    "published": result.published.isoformat(),
                    "pdf_url": result.pdf_url,
                    "entry_id": result.entry_id
                }
                batch_papers.append(paper)
                if len(batch_papers) >= self.max_papers:
                    break
        except Exception as e:
            print(f"Error fetching papers: {e}")
            return []

        # Remove blacklisted or skip-listed papers
        clean = []
        for p in batch_papers:
            arxiv_id = self._extract_arxiv_id(p.get('entry_id', ''))
            if arxiv_id and (self.blacklist.contains(arxiv_id) or self.skip_list.contains(arxiv_id)):
                print(f"  Skipping {p['title']} (ID {arxiv_id})")
                continue
            clean.append(p)

        if not clean:
            return []

        # Filter with Claude
        relevant = filter_agent.filter(topic, clean, threshold=threshold)

        # Update skip list with low-relevance papers 
        relevant_ids = {self._extract_arxiv_id(p.get('entry_id', '')) for p in relevant}
        for p in clean:
            arxiv_id = self._extract_arxiv_id(p.get('entry_id', ''))
            if arxiv_id and arxiv_id not in relevant_ids:
                self.skip_list.add(arxiv_id)
                print(f"  Added low-relevance to skip list: {p['title']} (ID {arxiv_id})")

        if len(relevant) < target_count:
            print(f"Warning: only {len(relevant)} relevant papers found, less than target {target_count}.")
        return relevant[:target_count]

    @staticmethod
    def _extract_arxiv_id(entry_id: str) -> str:
        import re
        match = re.search(r'abs/(\d+\.\d+)(?:v\d+)?', entry_id)
        return match.group(1) if match else ""