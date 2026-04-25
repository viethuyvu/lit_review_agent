# reference_analyzer.py
import anthropic
import os
import json
import re
from dotenv import load_dotenv
load_dotenv()

class ReferenceAnalyzer:
    def __init__(self, online_discovery):
        self.online = online_discovery
        self.claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def identify_foundational_references(self, paper_title: str, references: list) -> list:
        """
        Uses Claude to select the most foundational references from a list of dicts.
        Each reference dict should have at least 'title' and optionally 'citation_count'.
        Returns a list of dicts with keys: 'title', 'reason'.
        """
        if not references:
            return []

        # Build a numbered list of reference titles (with citation counts if available)
        ref_text = ""
        for i, ref in enumerate(references[:25], 1):
            title = ref.get('title', 'Unknown')
            cites = ref.get('citation_count', 'N/A')
            ref_text += f"{i}. {title} (citations: {cites})\n"

        prompt = f"""You are a research assistant. The paper titled "{paper_title}" has the following references. Identify the 3-5 references that are most foundational or important to the paper's core contributions (the ones that likely had the biggest impact). For each, give a short reason.

References:
{ref_text}

Return JSON list with keys: "title", "reason". Only include the most important ones.
Example: [{{"title": "Attention is All You Need", "reason": "Introduced transformer architecture that this paper builds upon."}}]"""

        try:
            response = self.claude.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=800,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.content[0].text.strip()
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                important = json.loads(json_match.group(0))
            else:
                important = []
            return important
        except Exception as e:
            print(f"Claude analysis failed: {e}")
            return []