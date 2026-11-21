"""Small text utilities shared by embeddings, reranking, the offline agent and evals."""
from __future__ import annotations

import re

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not
now of off on once only or other our ours ourselves out over own same she should so some such than that the
their theirs them themselves then there these they this those through to too under until up very was we were
what when where which while who whom why will with would you your yours yourself yourselves
company policy atlas freight please tell know us may also any whats i'm im i've i'd don't can't
""".split())

_WORD = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def stem(w: str) -> str:
    """Deliberately tiny suffix stripper (keeps the project dependency-free)."""
    for suf in ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ied", "ily", "ly", "ers", "er", "ed",
                "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            if suf == "ly" and w[-3] not in "thkr":  # nightly, monthly, weekly, hourly - not supply/family
                continue
            if suf == "ily" and w[-4] in "m":  # family
                continue
            base = w[: -len(suf)] + ("y" if suf in ("ies", "ied", "ily") else "")
            # submitted -> submitt -> submit ; travelling -> travell -> travel
            if len(base) > 3 and base[-1] == base[-2] and base[-1] not in "aeiouls":
                base = base[:-1]
            if len(base) > 5 and base.endswith("ll") and suf in ("ing", "ed", "er", "ers", "ings"):
                base = base[:-1]
            return base
    return w


def content_terms(text: str) -> list[str]:
    # single characters are kept on purpose: "Band C", "Plan B"
    return [stem(w) for w in words(text) if w not in STOPWORDS]


# Question-intent words: they shape the question but rarely appear in the answer
# passage ("how far in advance" vs "at least 10 days before"). Down-weighted, not dropped.
SOFT_TERMS = {stem(w) for w in """far advance long much many often quickly soon minimum maximum deadline need get
use using allowed happen time rate rule rules what which name process procedure apply applies standard
must given give do does done doing""".split()}


def sentences(text: str) -> list[str]:
    parts = []
    for block in re.split(r"\n\s*\n|\n(?=[-*•]|\d+[.)]\s)|\s(?=- [A-Z])", text):
        block = " ".join(block.split())
        if block:
            parts.extend(s.strip() for s in _SENT.split(block) if s.strip())
    return parts


# Domain thesaurus used by the query refiner. In the Claude path the model rewrites
# queries itself; this keeps the offline agent honest about the same behaviour.
SYNONYMS: dict[str, list[str]] = {
    "vacation": ["annual leave", "paid time off"],
    "holiday": ["annual leave", "public holidays"],
    "pto": ["annual leave", "paid time off"],
    "salary": ["pay", "compensation", "pay band"],
    "pay": ["salary", "compensation"],
    "wfh": ["remote work", "work from home"],
    "home": ["remote work"],
    "laptop": ["device", "equipment"],
    "fired": ["dismissal", "termination"],
    "fire": ["dismissal", "termination"],
    "sick": ["sick leave", "medical certificate"],
    "baby": ["maternity leave", "paternity leave"],
    "maternity": ["parental leave"],
    "paternity": ["parental leave"],
    "hack": ["security incident", "data breach"],
    "breach": ["data breach", "security incident"],
    "lawsuit": ["litigation", "legal hold"],
    "sue": ["litigation", "legal hold"],
    "contract": ["vendor agreement", "clause"],
    "trip": ["travel", "business travel"],
    "flight": ["air travel", "travel"],
    "hotel": ["accommodation", "lodging"],
    "meal": ["per diem", "meals"],
    "gift": ["gifts", "hospitality"],
    "bonus": ["variable pay", "incentive"],
    "password": ["passphrase", "credentials"],
    "phishing": ["suspicious email", "security incident"],
    "harassment": ["code of conduct", "misconduct"],
    "complaint": ["grievance", "report"],
    "warning": ["disciplinary", "written warning"],
    "overtime": ["working hours", "overtime"],
    "hours": ["working hours", "schedule"],
    "insurance": ["health insurance", "medical coverage"],
    "training": ["learning budget", "professional development"],
    "pregnant": ["maternity leave"],
    "off": ["leave"],
    "expense": ["reimbursed", "expense claim"],
    "minibar": ["minibar charges"],
    "length": ["characters long"],
    "minimum": ["at least"],
    "mileage": ["per kilometre"],
}


def term_alternatives(term: str) -> list[frozenset[str]]:
    """Ways a stemmed query term can be satisfied: the term itself, or ALL the terms of
    one of its synonym phrases ("pto" <- {"annual", "leave"} but not {"time"} alone)."""
    alts = [frozenset({term})]
    for raw, syns in SYNONYMS.items():
        if stem(raw) == term:
            alts += [frozenset(content_terms(syn)) for syn in syns if content_terms(syn)]
    return alts


def satisfied(term_alts: list[frozenset[str]], terms: set[str]) -> bool:
    return any(a <= terms for a in term_alts)


def expand_query(q: str) -> str:
    extra: list[str] = []
    for w in words(q):
        extra.extend(SYNONYMS.get(w, []))
    return q if not extra else q + " " + " ".join(dict.fromkeys(extra))
