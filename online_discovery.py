# online_discovery.py
import requests
import re
import time
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
load_dotenv()

class OnlineDiscovery:
    def __init__(self, delay=0.5):
        self.delay = delay
        self.cache = {}
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.headers = {}
        if self.api_key:
            self.headers['x-api-key'] = self.api_key
            print("Using Semantic Scholar API key (higher rate limits).")
        else:
            print("No Semantic Scholar API key found. Rate limits will be lower. Get a free key at semanticscholar.org.")

    def _make_request(self, url: str) -> Optional[Dict]:
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                print("Rate limit exceeded. Waiting 2 seconds...")
                time.sleep(2)
                return self._make_request(url)
            else:
                print(f"API error {resp.status_code}: {resp.text[:100]}")
                return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None

    def _arxiv_id_to_s2_id(self, arxiv_id: str) -> Optional[str]:
        clean_id = re.sub(r'v\d+$', '', arxiv_id)
        url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{clean_id}?fields=paperId"
        data = self._make_request(url)
        if data:
            return data.get('paperId')
        return None

    def get_paper_metadata(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch full metadata for a paper using its arXiv ID.
        Returns dict with keys: s2_id, title, authors, abstract, citationCount,
        referenceCount, influentialCitationCount, openAccessPdf, url, externalIds.
        """
        s2_id = self._arxiv_id_to_s2_id(arxiv_id)
        if not s2_id:
            return None
        # Removed 'doi' from fields to avoid API error
        url = f"https://api.semanticscholar.org/graph/v1/paper/{s2_id}?fields=title,authors,abstract,citationCount,referenceCount,influentialCitationCount,openAccessPdf,url,externalIds"
        data = self._make_request(url)
        if not data:
            return None
        authors = [a.get('name', '') for a in data.get('authors', [])]
        return {
            's2_id': s2_id,
            'title': data.get('title', ''),
            'authors': authors,
            'abstract': data.get('abstract', ''),
            'citation_count': data.get('citationCount', 0),
            'reference_count': data.get('referenceCount', 0),
            'influential_citation_count': data.get('influentialCitationCount', 0),
            'open_access_pdf': data.get('openAccessPdf', {}).get('url') if data.get('openAccessPdf') else None,
            'url': data.get('url')
        }

    def get_references(self, arxiv_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get structured list of references (papers cited by this paper)."""
        s2_id = self._arxiv_id_to_s2_id(arxiv_id)
        if not s2_id:
            return []
        url = f"https://api.semanticscholar.org/graph/v1/paper/{s2_id}?fields=references.title,references.authors,references.externalIds,references.url,references.citationCount,references.year"
        data = self._make_request(url)
        if not data:
            return []
        refs = data.get('references', [])
        results = []
        for ref in refs[:limit]:
            arxiv_id_ref = ref.get('externalIds', {}).get('ArXiv')
            authors = [a.get('name', '') for a in ref.get('authors', [])]
            results.append({
                'title': ref.get('title', 'Unknown'),
                'authors': authors,
                'arxiv_id': arxiv_id_ref,
                'url': ref.get('url'),
                'citation_count': ref.get('citationCount', 0),
                'year': ref.get('year')
            })
            time.sleep(self.delay)
        return results

    def get_citing_papers(self, arxiv_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get structured list of papers that cite this paper."""
        s2_id = self._arxiv_id_to_s2_id(arxiv_id)
        if not s2_id:
            return []
        url = f"https://api.semanticscholar.org/graph/v1/paper/{s2_id}?fields=citations.title,citations.authors,citations.externalIds,citations.url,citations.citationCount,citations.year"
        data = self._make_request(url)
        if not data:
            return []
        cites = data.get('citations', [])
        results = []
        for cit in cites[:limit]:
            arxiv_id_cit = cit.get('externalIds', {}).get('ArXiv')
            authors = [a.get('name', '') for a in cit.get('authors', [])]
            results.append({
                'title': cit.get('title', 'Unknown'),
                'authors': authors,
                'arxiv_id': arxiv_id_cit,
                'url': cit.get('url'),
                'citation_count': cit.get('citationCount', 0),
                'year': cit.get('year')
            })
            time.sleep(self.delay)
        return results

    def find_similar_papers(self, arxiv_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recommended similar papers via Semantic Scholar recommendations API."""
        s2_id = self._arxiv_id_to_s2_id(arxiv_id)
        if not s2_id:
            return []
        url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{s2_id}?fields=title,authors,externalIds,url,citationCount&limit={limit}"
        data = self._make_request(url)
        if not data:
            return []
        recommended = data.get('recommendedPapers', [])
        results = []
        for rec in recommended:
            arxiv_id_rec = rec.get('externalIds', {}).get('ArXiv')
            authors = [a.get('name', '') for a in rec.get('authors', [])]
            results.append({
                'title': rec.get('title', 'Unknown'),
                'authors': authors,
                'arxiv_id': arxiv_id_rec,
                'url': rec.get('url'),
                'citation_count': rec.get('citationCount', 0)
            })
            time.sleep(self.delay)
        return results

    def search_arxiv_by_title(self, title_query: str) -> Optional[str]:
        """Fallback: search arXiv by title to find arXiv ID."""
        import urllib.parse
        query_url = f"http://export.arxiv.org/api/query?search_query=ti:{urllib.parse.quote(title_query)}&start=0&max_results=1"
        try:
            resp = requests.get(query_url, timeout=10)
            if resp.status_code == 200:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(resp.text)
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                entry = root.find('atom:entry', ns)
                if entry is not None:
                    id_elem = entry.find('atom:id', ns)
                    if id_elem is not None:
                        match = re.search(r'abs/(\d+\.\d+)(?:v\d+)?', id_elem.text)
                        if match:
                            return match.group(1)
            return None
        except Exception as e:
            print(f"Title search failed: {e}")
            return None