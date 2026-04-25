# research_navigator.py
import os
import re
from download import DownloadAgent
from reader import ReaderAgent
from synthesis import SynthesisAgent
from database import PaperDatabase
from knowledge_extractor import KnowledgeExtractor
from vector_store import VectorStore
from blacklist import BlacklistManager
from skip_list import SkipListManager
from online_discovery import OnlineDiscovery
from search import SearchAgent
from reference_analyzer import ReferenceAnalyzer   # new, separate file for Claude analysis

import anthropic
from dotenv import load_dotenv
load_dotenv()

class ResearchNavigator:
    """
    Main orchestrator for the Research Navigator tool.
    Provides quick literature review, deep dive discovery, blacklist/skip list management,
    and discovery of highly cited papers not in the collection.
    """

    def __init__(self):
        self.db = PaperDatabase()
        self.vector_store = VectorStore()
        self.knowledge_extractor = KnowledgeExtractor()
        self.blacklist = BlacklistManager()
        self.skip_list = SkipListManager()
        self.online = OnlineDiscovery()
        self.search_agent = SearchAgent(self.blacklist, self.skip_list, max_papers=200)
        self.ref_analyzer = ReferenceAnalyzer(self.online)   # uses Claude
        self.claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _extract_arxiv_id(self, entry_id: str) -> str:
        """Extract clean arXiv ID (e.g., '2101.00101') from entry_id URL."""
        match = re.search(r'abs/(\d+\.\d+)(?:v\d+)?', entry_id)
        return match.group(1) if match else ""

    def _paper_exists(self, arxiv_id: str) -> bool:
        if not arxiv_id:
            return False
        return self.db.paper_exists(arxiv_id)

    def _get_output_dir(self) -> str:
        out_dir = "synthesis_output"
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def show_blacklist(self):
        bl = self.blacklist.list_all()
        if not bl:
            print("\nBlacklist is empty.")
            return
        print(f"\nBlacklist ({len(bl)} papers):")
        for idx, aid in enumerate(bl, 1):
            print(f"  {idx}. {aid}")
        if input("Clear all? (y/n): ").strip().lower() == 'y':
            self.blacklist.clear()
            print("Cleared.")

    def show_skip_list(self):
        sl = self.skip_list.list_all()
        if not sl:
            print("\nSkip list is empty.")
            return
        print(f"\nSkip list ({len(sl)} papers):")
        for idx, aid in enumerate(sl, 1):
            print(f"  {idx}. {aid}")
        if input("Clear all? (y/n): ").strip().lower() == 'y':
            self.skip_list.clear()
            print("Cleared.")

    def quick_review(self):
        """
        Mode 1: Quick Literature Review.
        - Fetches relevant papers (using SearchAgent, which filters via Claude).
        - Downloads PDFs, extracts summaries, knowledge, and stores in DB.
        - Enriches each new paper with Semantic Scholar metadata (citation counts, references count, etc.).
        - Generates a Markdown literature review.
        """
        topic = input("Enter research topic: ").strip()
        if not topic:
            topic = "transformers for time series forecasting"
        default_target = 13
        target_input = input(f"Number of relevant papers to fetch (default {default_target}): ").strip()
        target = int(target_input) if target_input.isdigit() else default_target
        print(f"\nSearching arXiv for: {topic} (target {target} relevant papers)")
        relevant = self.search_agent.fetch_relevant_papers(topic, target, threshold=6.0)
        if not relevant:
            print("No relevant papers found.")
            return
        print(f"Retained {len(relevant)} papers.")
        for i, p in enumerate(relevant, 1):
            print(f"{i}. {p['title']} (score: {p['relevance_score']}/10)")

        print("\nDownloading PDFs...")
        download_agent = DownloadAgent(cache_dir="pdf_cache")
        download_results = []
        for paper in relevant:
            success, path, msg = download_agent.download(paper)
            download_results.append({"title": paper["title"], "success": success, "local_path": path})
            print(f"  {msg}")
            if not success:
                aid = self._extract_arxiv_id(paper.get('entry_id', ''))
                if aid:
                    self.blacklist.add(aid)

        paper_summaries = []
        reader_agent = ReaderAgent()
        for paper, dl in zip(relevant, download_results):
            if not (dl["success"] and dl["local_path"]):
                print(f"Skipping {paper['title']} (download failed)")
                continue
            aid = self._extract_arxiv_id(paper.get("entry_id", ""))
            if self._paper_exists(aid):
                print(f"Using cached: {paper['title']}")
                existing = self.db.get_paper(aid)
                if existing and 'summaries' in existing:
                    paper_summaries.append(existing['summaries'])
                else:
                    summary = reader_agent.process_paper(dl["local_path"], paper["title"], paper["authors"])
                    paper_summaries.append(summary)
                    extracted = self.knowledge_extractor.extract_from_summary(summary)
                    self.db.add_paper(aid, paper["title"], paper["authors"], paper["summary"],
                                      dl["local_path"], summary, extracted)
                    self.vector_store.add_paper({"arxiv_id": aid, "title": paper["title"], "authors": paper["authors"],
                                                 "abstract": paper["summary"], "pdf_path": dl["local_path"],
                                                 "summaries": summary, "extracted": extracted})
                continue

            # New paper – full processing + Semantic Scholar enrichment
            print(f"Reading: {paper['title'][:60]}...")
            summary = reader_agent.process_paper(dl["local_path"], paper["title"], paper["authors"])
            paper_summaries.append(summary)
            print(f"Extracting knowledge from {paper['title'][:50]}...")
            extracted = self.knowledge_extractor.extract_from_summary(summary)

            # Fetch Semantic Scholar metadata
            print(f"Fetching Semantic Scholar metadata for {paper['title'][:50]}...")
            s2_meta = self.online.get_paper_metadata(aid)
            if s2_meta:
                s2_id = s2_meta.get('s2_id')
                citation_count = s2_meta.get('citation_count', 0)
                reference_count = s2_meta.get('reference_count', 0)
                influential_citation_count = s2_meta.get('influential_citation_count', 0)
                doi = s2_meta.get('doi')
                open_access_pdf = s2_meta.get('open_access_pdf')
            else:
                s2_id = None
                citation_count = reference_count = influential_citation_count = 0
                doi = open_access_pdf = None
                print("  (Could not retrieve Semantic Scholar data)")

            paper_dict = {
                "arxiv_id": aid,
                "title": paper["title"],
                "authors": paper["authors"],
                "abstract": paper["summary"],
                "pdf_path": dl["local_path"],
                "summaries": summary,
                "extracted": extracted
            }
            self.db.add_paper(
                arxiv_id=aid,
                title=paper["title"],
                authors=paper["authors"],
                abstract=paper["summary"],
                pdf_path=dl["local_path"],
                summaries=summary,
                extracted=extracted,
                s2_id=s2_id,
                citation_count=citation_count,
                reference_count=reference_count,
                influential_citation_count=influential_citation_count,
                doi=doi,
                open_access_pdf=open_access_pdf
            )
            self.vector_store.add_paper(paper_dict)

        print("\nSynthesising literature review in Markdown...")
        synthesis_agent = SynthesisAgent()
        md = synthesis_agent.synthesize(topic, paper_summaries)
        out_dir = self._get_output_dir()
        safe = topic.replace(' ', '_').replace('/', '_')
        fname = os.path.join(out_dir, f"literature_review_{safe}.md")
        with open(fname, "w") as f:
            f.write(md)
        print(f"Markdown saved to {fname}\nQuick review complete. {len(paper_summaries)} papers stored.\n")

    def deep_dive(self):
        """
        Mode 2: Deep Dive on a selected paper.
        - Lists papers from the database.
        - Offers online discovery: similar papers, citing papers, references (via API),
          and analysis of key references using Claude (via ReferenceAnalyzer).
        """
        papers = self.db.get_all_papers()
        if not papers:
            print("No papers in database. Run Quick Literature Review first.")
            return
        print("\nPapers in database:")
        for idx, p in enumerate(papers, 1):
            cite_info = f" (citations: {p.get('citation_count', 0)})" if p.get('citation_count') else ""
            print(f"{idx}. {p['title']}{cite_info}")
        choice = input("Select paper number (0 to cancel): ").strip()
        if not choice.isdigit() or int(choice) == 0:
            return
        idx = int(choice) - 1
        if idx < 0 or idx >= len(papers):
            print("Invalid.")
            return
        paper = papers[idx]
        arxiv_id = paper.get('arxiv_id')
        if not arxiv_id:
            print("No arXiv ID for this paper. Cannot perform online discovery.")
            return

        while True:
            print(f"\n--- Online Discovery: {paper['title'][:60]} ---")
            print("1. Find similar papers online")
            print("2. Show papers that cite this (online)")
            print("3. Show references (what it cites) - from Semantic Scholar")
            print("4. Analyze key references (Claude) - using Semantic Scholar data")
            print("5. Back to main menu")
            sub = input("Your choice: ").strip()
            if sub == "1":
                sim = self.online.find_similar_papers(arxiv_id, limit=10)
                if not sim:
                    print("No similar papers found.")
                else:
                    print("\nSimilar papers (Semantic Scholar recommendations):")
                    for s in sim:
                        aid = s['arxiv_id'] or "no arXiv ID"
                        print(f"  - {s['title']} (arXiv:{aid}, citations: {s.get('citation_count', '?')})")
            elif sub == "2":
                citing = self.online.get_citing_papers(arxiv_id, limit=15)
                if not citing:
                    print("No citing papers found.")
                else:
                    print("\nPapers that cite this:")
                    for c in citing:
                        aid = c['arxiv_id'] or "no arXiv ID"
                        print(f"  - {c['title']} (arXiv:{aid}, citations: {c.get('citation_count', '?')}, year: {c.get('year', '?')})")
            elif sub == "3":
                refs = self.online.get_references(arxiv_id, limit=20)
                if not refs:
                    print("No references found.")
                else:
                    print("\nReferences (papers cited by this):")
                    for r in refs:
                        aid = r['arxiv_id'] or "no arXiv ID"
                        print(f"  - {r['title']} (arXiv:{aid}, citations: {r.get('citation_count', '?')}, year: {r.get('year', '?')})")
            elif sub == "4":
                refs = self.online.get_references(arxiv_id, limit=30)
                if not refs:
                    print("No references found.")
                else:
                    # Use the separate ReferenceAnalyzer module
                    important = self.ref_analyzer.identify_foundational_references(paper['title'], refs)
                    if not important:
                        print("No foundational references identified by Claude.")
                    else:
                        print("\nFoundational / Key References (Claude analysis):")
                        for i, ref in enumerate(important, 1):
                            print(f"{i}. {ref.get('title', 'Unknown')}")
                            if ref.get('reason'):
                                print(f"   Reason: {ref['reason']}")
                            # Optionally attempt to find arXiv ID for the reference title
                            arxiv_id_ref = self.online.search_arxiv_by_title(ref['title'])
                            if arxiv_id_ref:
                                print(f"   arXiv:{arxiv_id_ref} - https://arxiv.org/abs/{arxiv_id_ref}")
                            print()
            elif sub == "5":
                break
            else:
                print("Invalid option.")

    def discover_highly_cited_papers(self):
        """
        Mode 5: Find highly cited papers (citation count > threshold) not in your collection.
        Uses Semantic Scholar search with a keyword.
        """
        print("\n--- Discover Highly Cited Papers ---")
        threshold = input("Minimum citation count (default 100): ").strip()
        threshold = int(threshold) if threshold.isdigit() else 100
        keyword = input("Enter a keyword to search (e.g., 'machine learning'): ").strip()
        if not keyword:
            print("No keyword provided. Aborting.")
            return

        import urllib.parse
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": keyword,
            "fields": "title,authors,externalIds,url,citationCount,paperId",
            "limit": 50,
            "offset": 0
        }
        found = []
        while len(found) < 50:
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            data = self.online._make_request(url)  # reuse the authenticated request
            if not data or not data.get('data'):
                break
            for paper in data['data']:
                cit_count = paper.get('citationCount', 0)
                if cit_count >= threshold:
                    arxiv_id = paper.get('externalIds', {}).get('ArXiv')
                    if arxiv_id and not self.db.paper_exists(arxiv_id):
                        found.append({
                            'title': paper.get('title', 'Unknown'),
                            'arxiv_id': arxiv_id,
                            'citation_count': cit_count,
                            'url': paper.get('url')
                        })
            params['offset'] += params['limit']
            if params['offset'] > 500:
                break
        if not found:
            print(f"No highly cited papers found for keyword '{keyword}' above threshold {threshold}.")
        else:
            print(f"\nFound {len(found)} highly cited papers not in your collection:")
            for i, p in enumerate(found[:20], 1):
                print(f"{i}. {p['title']} (arXiv:{p['arxiv_id']}, citations: {p['citation_count']})")