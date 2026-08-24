"""CV_SCHEMA — the structured shape every downstream stage reads.

After preprocessing, nothing reads the raw CV again. Stage 1, 2 and 3 all work from
this JSON (anonymized), which is why the schema is `strict: True` and every field is
required: a missing key downstream is a bug, never a "maybe the LLM omitted it".
"""

CV_SCHEMA = {
    "type": "object",
    "properties": {
        "full_name": {"type": "string"},
        "email": {"type": "string"},
        "phone": {"type": "string"},
        "languages": {
            "type": "array",
            "items": {"type": "object", "properties": {
                "language": {"type": "string"},
                "proficiency": {"type": "string", "enum": ["native", "fluent", "professional", "basic"]}
            }, "required": ["language", "proficiency"]}
        },
        "education": {
            "type": "array",
            "items": {"type": "object", "properties": {
                "degree": {"type": "string"},
                "field": {"type": "string"},
                "institution": {"type": "string"},
                "end_year": {"type": ["integer", "null"]}
            }, "required": ["degree", "field", "institution", "end_year"]}
        },
        "experience": {
            "type": "array",
            "items": {"type": "object", "properties": {
                "title": {"type": "string"},
                "company": {"type": "string"},
                "start_date": {"type": ["string", "null"]},
                "end_date": {"type": ["string", "null"]},
                "description": {"type": "string"},
                "evidence_quote": {"type": "string"}
            }, "required": ["title", "company", "start_date", "end_date", "description", "evidence_quote"]}
        },
        "skills": {"type": "array", "items": {"type": "string"}},
        "total_years_experience": {"type": "number"},
        "certifications": {"type": "array", "items": {"type": "string"}},
        "extraction_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "extraction_notes": {"type": "string"}
    },
    "required": ["full_name", "email", "phone", "languages", "education", "experience",
                 "skills", "total_years_experience", "certifications", "extraction_confidence", "extraction_notes"]
}
