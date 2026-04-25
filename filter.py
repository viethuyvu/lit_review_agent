import anthropic
from dotenv import load_dotenv
import os
import re

load_dotenv()

class FilterAgent:
    def __init__(self, model="claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def relevance_score(self, topic: str, title: str, abstract: str) -> float:
        prompt = f"""Topic: {topic}
Title: {title}
Abstract: {abstract}

On a scale of 0 to 10, how relevant is this paper to the topic?
Return ONLY a number between 0 and 10, nothing else."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        try:
            text = response.content[0].text.strip()
            match = re.search(r"(\d+(?:\.\d+)?)", text)
            score = float(match.group(1)) if match else 0.0
        except:
            score = 0.0
        return min(10.0, max(0.0, score))

    def filter(self, topic: str, papers: list, threshold=6.0):
        """Adds relevance_score to each paper and returns those with score >= threshold."""
        filtered = []
        for paper in papers:
            score = self.relevance_score(topic, paper["title"], paper["summary"])
            paper["relevance_score"] = score
            if score >= threshold:
                filtered.append(paper)
        return filtered