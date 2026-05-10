"""
context_extractor.py
--------------------
Extracts a structured "hiring context" from the full conversation history.
Called on every POST /chat request (stateless — rebuilt from scratch each time).
"""

import json
from typing import Optional
from groq import Groq

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class HiringContext:
    """
    Everything we know about what the user wants.
    All fields are Optional — we fill in what we can extract.
    """

    def __init__(self):
        self.role:             Optional[str]       = None   # e.g. "Java developer"
        self.seniority:        Optional[str]       = None   # e.g. "mid-level"
        self.years_exp:        Optional[int]       = None   # e.g. 4
        self.skills:           list[str]           = []     # e.g. ["Java", "stakeholder mgmt"]
        self.test_types:       list[str]           = []     # e.g. ["Knowledge & Skills", "Personality & Behavior"]
        self.duration_max_min: Optional[int]       = None   # max duration in minutes
        self.languages:        list[str]           = []     # e.g. ["English (USA)"]
        self.remote_only:      Optional[bool]      = None
        self.industry:         Optional[str]       = None   # e.g. "fintech", "healthcare"
        self.additional_notes: Optional[str]       = None

    def to_dict(self) -> dict:
        return {
            "role":             self.role,
            "seniority":        self.seniority,
            "years_exp":        self.years_exp,
            "skills":           self.skills,
            "test_types":       self.test_types,
            "duration_max_min": self.duration_max_min,
            "languages":        self.languages,
            "remote_only":      self.remote_only,
            "industry":         self.industry,
            "additional_notes": self.additional_notes,
        }

    def has_enough_to_recommend(self) -> bool:
        """
        Minimum bar: we know the role.
        Ideally we also have seniority, but role alone is enough for 1 turn.
        """
        return self.role is not None

    def to_search_query(self) -> str:
        """
        Build a semantic search query string from what we know.
        """
        parts = []
        if self.role:
            parts.append(self.role)
        if self.skills:
            parts.append(", ".join(self.skills))
        if self.seniority:
            parts.append(f"{self.seniority} level")
        if self.years_exp:
            parts.append(f"{self.years_exp} years experience")
        if self.industry:
            parts.append(self.industry)
        if self.test_types:
            parts.append(", ".join(self.test_types))
        if self.additional_notes:
            parts.append(self.additional_notes)
        return " ".join(parts)

    def get_job_level_filters(self) -> Optional[list[str]]:
        """
        Map seniority string → SHL job level values.
        Returns None if we cannot map (= no filter applied).
        """
        if not self.seniority:
            return None

        s = self.seniority.lower()

        mapping = {
            "entry":        ["Entry-Level"],
            "junior":       ["Entry-Level", "Graduate"],
            "graduate":     ["Graduate", "Entry-Level"],
            "mid":          ["Mid-Professional", "Professional Individual Contributor"],
            "senior":       ["Professional Individual Contributor", "Mid-Professional"],
            "lead":         ["Professional Individual Contributor", "Manager", "Front Line Manager"],
            "manager":      ["Manager", "Front Line Manager", "Supervisor"],
            "director":     ["Director", "Manager"],
            "executive":    ["Executive", "Director"],
            "vp":           ["Executive", "Director"],
        }

        for key, levels in mapping.items():
            if key in s:
                return levels

        return None

    def get_key_filters(self) -> Optional[list[str]]:
        """
        Map user-expressed test type preferences → SHL keys.
        Returns None if no preference expressed.
        """
        if not self.test_types:
            return None
        return self.test_types


# ---------------------------------------------------------------------------
# SYSTEM PROMPT FOR EXTRACTION
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """
You are a structured data extractor for an SHL assessment recommendation system.

Given a conversation between a hiring manager and an AI assistant, extract hiring intent.

Return ONLY valid JSON with this exact schema (use null for unknown fields):
{
  "role": string or null,
  "seniority": string or null,
  "years_exp": integer or null,
  "skills": [list of strings],
  "test_types": [list from: "Knowledge & Skills", "Personality & Behavior", "Simulations", "Competencies", "Biodata & Situational Judgment", "Assessment Exercises", "Ability & Aptitude", "Development & 360"],
  "duration_max_min": integer or null,
  "languages": [list of strings],
  "remote_only": boolean or null,
  "industry": string or null,
  "additional_notes": string or null
}

Rules:
- role: job title or function (e.g. "Java developer", "sales manager", "data scientist")
- seniority: one of "entry", "junior", "graduate", "mid", "senior", "lead", "manager", "director", "executive"
- skills: technical skills, tools, languages, soft skills mentioned
- test_types: infer from context. "personality" → "Personality & Behavior". "coding" → "Simulations". "cognitive" → "Ability & Aptitude". "knowledge" → "Knowledge & Skills"
- If user says "no preference" for something, leave it null
- Do not include markdown. Return only the JSON object.
"""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class ContextExtractor:

    def __init__(self, groq_client: Groq):
        self.client = groq_client

    def extract(self, messages: list[dict]) -> HiringContext:
        """
        Takes the full conversation history and returns a HiringContext.
        """
        # Format conversation for the LLM
        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in messages
        )

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Extract hiring context from this conversation:\n\n{conversation_text}"}
                ],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            data = json.loads(raw)

        except Exception as e:
            print(f"[ContextExtractor] Extraction failed: {e}")
            data = {}

        return self._dict_to_context(data)

    def _dict_to_context(self, data: dict) -> HiringContext:
        ctx = HiringContext()

        ctx.role             = data.get("role")
        ctx.seniority        = data.get("seniority")
        ctx.years_exp        = data.get("years_exp")
        ctx.skills           = data.get("skills") or []
        ctx.test_types       = data.get("test_types") or []
        ctx.duration_max_min = data.get("duration_max_min")
        ctx.languages        = data.get("languages") or []
        ctx.remote_only      = data.get("remote_only")
        ctx.industry         = data.get("industry")
        ctx.additional_notes = data.get("additional_notes")

        return ctx