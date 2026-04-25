# synthesis_agent.py
import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

class SynthesisAgent:
    def __init__(self, model="claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def synthesize(self, topic: str, paper_summaries: list) -> str:
        if not paper_summaries:
            return self._empty_markdown(topic)

        # Build compact representation of papers
        papers_text = ""
        for i, paper in enumerate(paper_summaries, 1):
            papers_text += f"\n## Paper {i}: {paper['title']}\n\n"
            papers_text += f"**Authors:** {', '.join(paper['authors'])}\n\n"
            for section, summary in paper['summaries'].items():
                if summary and "not found" not in summary.lower():
                    papers_text += f"**{section.capitalize()}:**\n\n{summary}\n\n"

        # Build reference list
        ref_list = "\n## References\n\n"
        for i, paper in enumerate(paper_summaries, 1):
            authors = ', '.join(paper['authors'])
            ref_list += f"{i}. {authors}. \"{paper['title']}\" (arXiv preprint).\n"

        prompt = f"""You are an expert research assistant. Write a thematic literature review on the topic:

**Topic:** {topic}

Below are summaries of {len(paper_summaries)} academic papers.

{papers_text}

**Instructions:**
Produce a Markdown document with the following structure:

# Literature Review: {topic}

## Introduction

[Write a clear introduction that sets the context and importance of the topic.]

## Major Approaches

[Group papers by methodology or theme. Compare and contrast.]

## Key Findings

[Summarise the most important results across papers.]

## Contradictions or Debates

[If papers disagree or highlight trade-offs, discuss them.]

## Common Limitations

[Synthesise limitations mentioned across papers.]

## Research Gaps

[Based on future work sections, identify open problems.]

## Conclusion

[Summarise the state of the field and suggest promising directions.]

{ref_list}

**Important rules:**
- Do not use in-text citations (no [@author], no numbered references inside the text).
- Write in academic English.
- Use bullet lists with - or * where appropriate.
- Output only the Markdown source, no extra commentary.

Now produce the Markdown document:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}]
            )
            md_code = response.content[0].text.strip()
            # Remove markdown code fences if present
            if md_code.startswith('```markdown'):
                md_code = md_code[11:]
            elif md_code.startswith('```'):
                md_code = md_code[3:]
            if md_code.endswith('```'):
                md_code = md_code[:-3]
            md_code = md_code.strip()

            # Ensure references are present
            if "## References" not in md_code:
                md_code += f"\n\n{ref_list}"
            return md_code
        except Exception as e:
            return self._error_markdown(topic, str(e))

    def _empty_markdown(self, topic):
        return f"# Literature Review: {topic}\n\nNo papers were summarised. Please try a different topic."

    def _error_markdown(self, topic, error_msg):
        return f"# Literature Review: {topic}\n\nSynthesis failed: {error_msg}"