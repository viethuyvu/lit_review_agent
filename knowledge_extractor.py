# knowledge_extractor.py
import anthropic
import json
import re
from dotenv import load_dotenv
import os

load_dotenv()

class KnowledgeExtractor:
    def __init__(self, model="claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def extract_from_summary(self, paper_summary: dict) -> dict:
        """
        paper_summary: dict from reader_agent.process_paper()
        Expected keys: 'title', 'authors', 'pdf_path', 'summaries'
        where 'summaries' is a dict with sections like 'introduction', 'methods', 'results', 'conclusion', etc.
        """
        title = paper_summary.get('title', 'Unknown')
        summaries = paper_summary.get('summaries', {})
        
        # Combine the most relevant sections for extraction
        combined_text = ""
        for section in ['introduction', 'methods', 'results', 'conclusion']:
            if section in summaries and summaries[section]:
                combined_text += f"\n=== {section.upper()} ===\n{summaries[section]}\n"
        
        if not combined_text.strip():
            return self._empty_extraction("No summary text available")
        
        # Truncate to avoid token limits (Claude Haiku 4.5 has 200k context, but keep safe)
        if len(combined_text) > 12000:
            combined_text = combined_text[:12000] + "..."
        
        prompt = f"""You are an AI research assistant. Extract structured information from the following paper summary.

Paper title: {title}

Paper summary (by sections):
{combined_text}

Extract the following fields as JSON. If a field is not present or cannot be determined, use an empty list or empty string as appropriate.

Fields:
- "methods": list of strings – specific techniques, algorithms, architectures, or approaches used (e.g., ["ProbSparse attention", "distilling encoder"]).
- "datasets": list of strings – datasets used for experiments (e.g., ["ETT", "Electricity", "Weather"]).
- "metrics": list of strings – evaluation metrics reported (e.g., ["MSE", "MAE", "RMSE"]).
- "key_findings": list of strings – 3-5 most important results or conclusions (e.g., ["4x speedup over standard Transformer", "SOTA on 6 benchmarks"]).
- "research_gaps": list of strings – limitations or future work mentioned (e.g., ["multivariate extension needed", "interpretability not evaluated"]).

Return ONLY a valid JSON object, no extra text. Example format:
{{
  "methods": ["method A", "method B"],
  "datasets": ["dataset1"],
  "metrics": ["accuracy", "F1"],
  "key_findings": ["finding1", "finding2"],
  "research_gaps": ["gap1"]
}}"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            result_text = response.content[0].text.strip()
            # Extract JSON from possible markdown or stray text
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(0)
            extracted = json.loads(result_text)
            # Validate required keys
            required_keys = ["methods", "datasets", "metrics", "key_findings", "research_gaps"]
            for key in required_keys:
                if key not in extracted:
                    extracted[key] = []
            return extracted
        except Exception as e:
            print(f"Knowledge extraction failed for '{title}': {e}")
            return self._empty_extraction(str(e))

    def _empty_extraction(self, error_msg=""):
        """Return an empty extraction structure."""
        return {
            "methods": [],
            "datasets": [],
            "metrics": [],
            "key_findings": [],
            "research_gaps": [],
            "_error": error_msg
        }