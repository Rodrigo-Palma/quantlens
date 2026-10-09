"""Output guardrails for generated explanations (English and Brazilian Portuguese).

Keep explanations descriptive: never let the model emit investment advice or a
guarantee. The API rejects an LLM output that trips any rule and serves the
deterministic explainer instead.

Text is lowercased and accent-stripped before matching, so ``"COMPRE JÁ"`` and
``"compre ja"`` are the same. Every pattern is anchored on word boundaries.
Rules marked negatable are skipped when a negation (``not``, ``no``, ``zero``,
``não``, ``nenhum``...) appears shortly before the match in the same clause, so
``"returns are not guaranteed"`` passes and ``"guaranteed returns"`` does not.
Directive rules are never negated: "you should not buy" is still advice.

This is a lexical filter with measured recall and false-positive rates (see
``quantlens.evals.guardrail``); it misses paraphrases it has no pattern for.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_NEGATION_WINDOW = 4
_NEGATIONS = frozenset(
    {"not", "no", "nothing", "never", "none", "nor", "neither", "zero", "nao", "nem"}
    | {"nenhum", "nenhuma", "nunca", "jamais", "qualquer"}
)
_CLAUSE_START = r"(?:^|[.;:!?,]\s*|\b(?:and|so|e|entao)\s+)(?:just\s+|please\s+)?"

_EN_SUBJECT = r"(?:you(?: guys| all)?|investors|traders|clients|holders|one)"
_EN_ACTIONS = (
    r"(?:buy|sell|accumulate|short|exit|dump|invest|add to|load up|take profits?|hold"
    r"|enter|go long|get (?:in|out))"
)
_PT_ACTIONS = (
    r"(?:comprar|vender|acumular|zerar|sair|entrar|investir|aumentar|reduzir|realizar"
    r"|liquidar)"
)
_PT_IMPERATIVES = (
    r"(?:compre|compra|venda(?!\s+(?:coberta|de put))|vende|zere|monte|entre|entra|realize"
    r"|saia|sai|acumule|acumula|coloque|aproveite|aproveita|aposte|aposta|invista|investe"
    r"|segure|segura|mantenha|mantem|faca preco medio|pode comprar)"
)

# (rule id, pattern, negatable)
_RULES: tuple[tuple[str, str, bool], ...] = (
    # English: directives and recommendations.
    (
        "en.directive",
        rf"\b{_EN_SUBJECT}\s+(?:\w+\s+)?(?:should|must|ought to|need to|have to)"
        rf"(?:\s+not)?\s+(?:\w+\s+)?{_EN_ACTIONS}\b",
        False,
    ),
    ("en.directive", rf"\b(?:smart|wise|time)\s+to\s+{_EN_ACTIONS}\b", False),
    (
        "en.imperative",
        _CLAUSE_START + r"(?:buy|sell|short|exit|dump|grab|average down|go long|go all in"
        r"|load up|take profits?|get in(?:to)?|act now|back up the truck|hold on(?:to| to)"
        r"|hold your|keep holding|start (?:buying|selling|accumulating)|accumulate|enter)\b(?!-)",
        False,
    ),
    ("en.imperative", r"\b(?:buy|sell)\s+(?:now|today|immediately|the dip)\b", False),
    ("en.imperative", r"\b(?:consider|start)\s+(?:adding|buying|selling|trimming)\b", False),
    ("en.imperative", r"\b(?:don'?t|do not)\s+(?:even\s+think\s+about\s+\w+ing|miss)\b", False),
    ("en.imperative", r"\bmaintain\s+(?:a\s+)?hold\b", False),
    (
        "en.recommend",
        r"\b(?:i|we|i'd|we'd)\s+(?:would\s+)?(?:strongly\s+)?(?:recommend|advise|suggest)\b",
        False,
    ),
    ("en.recommend", r"\badvises?\s+(?:you|clients|investors)\b", False),
    ("en.recommend", r"\bour\s+(?:recommendation|advice|call|guidance)\b", False),
    (
        "en.recommend",
        r"\brecommendation\s+(?:is\s+)?to\s+(?:buy|sell|hold|exit|enter|accumulate)\b",
        True,
    ),
    (
        "en.recommend",
        r"\b(?:recommendation|rating|rated)\s*:\s*(?:strong\s+)?(?:buy|sell)\b",
        False,
    ),
    ("en.recommend", r"\bstrong\s+(?:buy|sell)\b(?!-)", False),
    ("en.recommend", r"\bi\s+would\s+(?:buy|sell)\b", False),
    (
        "en.recommend",
        r"\b(?:great|good|perfect|ideal|right)\s+(?:time|moment|entry point|entry)"
        r"\s+to\s+(?:buy|sell|enter|exit|get in)\b",
        False,
    ),
    ("en.recommend", r"\b(?:buying|selling)\s+(?:here|now|at these|at this)\b", False),
    ("en.recommend", r"\bno-?brainer\b|\bcan'?t-?miss\b|\ball (?:of )?your savings\b", False),
    ("en.recommend", r"\block in (?:the |those |your )?(?:gains|profits)\b", False),
    # English: guarantees and certainty.
    ("en.guarantee", r"\bguarantee[sd]?\b", True),
    (
        "en.guarantee",
        r"\brisk-?free\b(?!\s+rate)|\bzero risk\b|\bno risk\b(?!-)|\bwithout risk\b",
        True,
    ),
    ("en.guarantee", r"\b(?:can'?t|cannot|can not)\s+(?:lose|fail|go wrong)\b", False),
    ("en.guarantee", r"\bwon'?t lose\b|\bi promise\b", False),
    ("en.guarantee", r"\bsure (?:thing|win)\b|\bwill (?:definitely|surely|certainly)\b", True),
    ("en.guarantee", r"\bdefinitely going to\b|\bis a lock\b", True),
    ("en.guarantee", r"\bcertain to\b|\bbound to (?:rise|go up|double|recover)\b", True),
    # Portuguese: directives and recommendations.
    (
        "pt.directive",
        rf"\b(?:deve|deveria|deveriam|devem|precisa|precisam)\s+{_PT_ACTIONS}\b",
        False,
    ),
    ("pt.directive", rf"\b(?:hora|momento(?:\s+\w+)?)\s+(?:de|para)\s+{_PT_ACTIONS}\b", False),
    ("pt.imperative", _CLAUSE_START + _PT_IMPERATIVES + r"\b", False),
    ("pt.imperative", r"\bcompre\b|\bnao perca\b|\bbora\b|\bda stop\b", False),
    ("pt.imperative", r"\bpode (?:comprar|vender|entrar|botar|investir|colocar)\b", False),
    (
        "pt.recommend",
        r"\b(?:recomendo|sugiro|indico|aconselho|recomendamos|sugerimos|aconselhamos"
        r"|conselhamos|orientamos|indicamos)\b",
        False,
    ),
    ("pt.recommend", r"\bminha\s+recomendacao\b|\brecomendacao\s*:", False),
    ("pt.recommend", r"\brecomendacao\b[^.;:!?]{0,40}?\b(?:compra|venda)\b", True),
    ("pt.recommend", r"\b(?:compra|venda)\s+(?:recomendada|certa)\b", False),
    ("pt.recommend", r"\boportunidade\s+de\s+(?:compra|venda)\b", False),
    ("pt.recommend", rf"\bvale\s+(?:muito\s+)?a\s+pena\s+{_PT_ACTIONS}\b", False),
    ("pt.recommend", r"\bembolsar\b|\btodas\s+as\s+suas\s+economias\b|\bsem medo\b", False),
    (
        "pt.recommend",
        r"\bmanutencao da posicao\b|\be imperativo\b|\bexige a (?:compra|venda)\b",
        False,
    ),
    (
        "pt.recommend",
        r"\b(?:unica|melhor)\s+(?:via|jogada|saida|entrada|opcao)\b|\bmedida mais prudente\b",
        False,
    ),
    # Portuguese: guarantees and certainty.
    ("pt.guarantee", r"\bgarantid[oa]s?\b|\bgarant(?:e|em|ir|imos)\b", True),
    ("pt.guarantee", r"\bassegur(?:a|ar|amos)\b|\binevitavel\b", True),
    ("pt.guarantee", r"\bsem riscos?\b|\brisco zero\b|\b(?:livre|isento) de riscos?\b", True),
    ("pt.guarantee", r"\bnao tem (?:como (?:perder|cair|dar errado)|erro)\b", False),
    (
        "pt.guarantee",
        r"\bcom certeza\b|\bcerteza (?:de|absoluta)\b|\b(?:ganho|lucro) certo\b",
        True,
    ),
    ("pt.guarantee", r"\bdinheiro facil\b", False),
)

_COMPILED = tuple((rule, re.compile(pattern), negatable) for rule, pattern, negatable in _RULES)
_CLAUSE_SPLIT = re.compile(r"[.;:!?\n]")
_WORD = re.compile(r"[a-z']+")
_CURLY_APOSTROPHE = "\N{RIGHT SINGLE QUOTATION MARK}"


@dataclass
class GuardrailResult:
    ok: bool
    violations: list[str] = field(default_factory=list)


def normalize(text: str) -> str:
    """Lowercase, strip accents and unify apostrophes."""
    decomposed = unicodedata.normalize("NFKD", text.replace(_CURLY_APOSTROPHE, "'"))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def _is_negated(text: str, start: int) -> bool:
    clause = _CLAUSE_SPLIT.split(text[:start])[-1]
    window = _WORD.findall(clause)[-_NEGATION_WINDOW:]
    return any(word in _NEGATIONS or word.endswith("n't") for word in window)


def validate(explanation: str) -> GuardrailResult:
    """Flag investment-advice or guarantee phrasing; violations name rule and text."""
    text = normalize(explanation)
    violations: list[str] = []
    for rule, pattern, negatable in _COMPILED:
        for match in pattern.finditer(text):
            if negatable and _is_negated(text, match.start()):
                continue
            violations.append(f"{rule}: {match.group(0).strip(' .;:!?,')}")
    return GuardrailResult(ok=not violations, violations=violations)
