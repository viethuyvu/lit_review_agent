import fitz  # pymupdf
import anthropic
from dotenv import load_dotenv
import os
import re

load_dotenv()

class ReaderAgent:
    def __init__(self, model="claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        # Common section headers (case-insensitive)
        self.section_headers = {
            "introduction": ["introduction", "intro", "background"],
            "methods": ["methods", "methodology", "approach", "experimental setup"],
            "results": ["results", "findings", "experiments"],
            "conclusion": ["conclusion", "conclusions", "discussion"],
            "limitations": ["limitations", "limitation"],
            "future_work": ["future work", "future directions", "future research"]
        }

    def extract_text_from_pdf(self, pdf_path):
        """Extract all text from a PDF file."""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def split_into_sections(self, text):
        """
        Split the full text into sections using heuristic header detection.
        Returns a dict: {section_name: section_text}
        """
        lines = text.split('\n')
        sections = {key: "" for key in self.section_headers.keys()}
        current_section = None
        current_text = []

        # Also capture "other" content that doesn't match any section
        other_text = []

        for line in lines:
            line_lower = line.lower().strip()
            matched = False
            for section, headers in self.section_headers.items():
                for header in headers:
                    # Match lines that look like section headers (e.g., "1. Introduction", "Methods")
                    pattern = r'^(\d+\.?\s*)?' + re.escape(header) + r'(\s*\d+)?$'
                    if re.match(pattern, line_lower):
                        # Save previous section content
                        if current_section and current_text:
                            sections[current_section] = ' '.join(current_text)
                        current_section = section
                        current_text = []
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                if current_section is not None:
                    current_text.append(line)
                else:
                    other_text.append(line)

        # Save the last section
        if current_section and current_text:
            sections[current_section] = ' '.join(current_text)

        # If no sections found, put everything in "other"
        if all(len(v) == 0 for v in sections.values()):
            sections["other"] = ' '.join(other_text)

        return sections

    def summarise_section(self, section_name, section_text, max_chars=3000):
        """
        Summarise a single section using Claude.
        Returns a summary string (3-5 bullet points).
        """
        if not section_text or len(section_text.strip()) < 50:
            return "No significant content found."

        # Truncate to avoid token limits
        if len(section_text) > max_chars:
            section_text = section_text[:max_chars] + "..."

        prompt = f"""You are a research assistant. Summarise the following {section_name} section of an academic paper.

{section_name.upper()} SECTION:
{section_text}

Provide a concise summary as 3-5 bullet points. Each bullet point should be a complete sentence.
Return ONLY the bullet points, one per line, starting with '- '.

Example:
- The paper proposes a new transformer architecture.
- Experiments show 15% improvement over baseline.
- The method is computationally efficient."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            summary = response.content[0].text.strip()
            # Ensure it starts with bullet points
            if not summary.startswith('-'):
                summary = '- ' + summary.replace('\n', '\n- ')
            return summary
        except Exception as e:
            return f"Summarisation failed: {str(e)}"

    def process_paper(self, pdf_path, title, authors):
        """
        Full pipeline for one paper: extract text, split into sections, summarise each.
        Returns a dict with paper metadata and section summaries.
        """
        print(f"Reading: {title[:60]}...")
        full_text = self.extract_text_from_pdf(pdf_path)
        sections = self.split_into_sections(full_text)

        summaries = {}
        for section_name, section_text in sections.items():
            if section_text and len(section_text.strip()) > 50:
                print(f"     Summarising {section_name}...")
                summaries[section_name] = self.summarise_section(section_name, section_text)
            else:
                summaries[section_name] = "Section not found or too short."

        return {
            "title": title,
            "authors": authors,
            "pdf_path": pdf_path,
            "summaries": summaries
        }