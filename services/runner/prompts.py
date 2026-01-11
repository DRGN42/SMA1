import json
from typing import Dict

ANALYSIS_SCHEMA = {
    "type": "object",
    "required": [
        "mood",
        "atmosphere",
        "tempo",
        "prosody_hints",
        "visual",
        "narration_text",
        "caption",
        "hashtags",
    ],
    "properties": {
        "mood": {"type": "string"},
        "atmosphere": {"type": "string"},
        "tempo": {"type": "string"},
        "prosody_hints": {"type": "string"},
        "visual": {
            "type": "object",
            "required": ["global_prompt", "style_prompt", "negative_prompt", "line_prompts"],
            "properties": {
                "global_prompt": {"type": "string"},
                "style_prompt": {"type": "string"},
                "negative_prompt": {"type": "string"},
                "line_prompts": {"type": "array", "items": {"type": "string"}},
            },
        },
        "narration_text": {"type": "string"},
        "caption": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
    },
}


def build_prompt(poem: Dict[str, str]) -> Dict[str, str]:
    schema_json = json.dumps(ANALYSIS_SCHEMA, ensure_ascii=False, indent=2)
    system = (
        "Du bist ein deutschsprachiger Creative Director für Poetry-Video-Produktionen. "
        "Du analysierst Gedichte und lieferst strukturierte JSON-Ausgaben für Audio, Visuals und Social Captioning. "
        "Antworte ausschließlich mit gültigem JSON gemäß Schema."
    )

    user = (
        "ANALYSE-Aufgabe:\n"
        "1) Analysiere Stimmung, Atmosphäre, Tempo, Prosodie.\n"
        "2) Erzeuge einen globalen Visual-Prompt, Style-Prompt und Negative-Prompt.\n"
        "3) Erzeuge line_prompts exakt in der Anzahl der Gedichtzeilen (ohne Leerzeilen).\n"
        "4) narration_text muss exakt der Gedichttext sein, ohne Zusatz.\n"
        "5) Caption + Hashtags (DE, plattformtauglich, nicht spammy).\n\n"
        f"Titel: {poem['title']}\n"
        f"Autor: {poem['author']}\n"
        f"Sprache: {poem.get('language', 'de')}\n"
        "Gedichttext:\n"
        f"{poem['text']}\n\n"
        "JSON Schema:\n"
        f"{schema_json}\n"
        "Antwort ausschließlich als JSON."
    )

    return {"system": system, "user": user}
