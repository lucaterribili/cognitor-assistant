"""
italian_pluralize.py
--------------------
Library for Italian language pluralization.
Handles grammatical rules, common exceptions, and invariable words.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class Gender(Enum):
    MASCULINE = "m"
    FEMININE = "f"
    INVARIABLE = "inv"
    UNKNOWN = "?"


@dataclass
class PluralResult:
    singular: str
    plural: str
    gender: Gender
    rule: str
    exception: bool = False


# ─── Irregular exceptions ───────────────────────────────────────────────────

EXCEPTIONS: dict[str, tuple[str, Gender]] = {
    # Masculine irregular
    "uomo":     ("uomini",   Gender.MASCULINE),
    "dio":      ("dei",      Gender.MASCULINE),
    "bue":      ("buoi",     Gender.MASCULINE),
    "tempio":   ("templi",   Gender.MASCULINE),
    "paio":     ("paia",     Gender.FEMININE),   # feminine plural!
    "centinaio":("centinaia",Gender.FEMININE),
    "migliaio": ("migliaia", Gender.FEMININE),
    "miglio":   ("miglia",   Gender.FEMININE),
    "braccio":  ("braccia",  Gender.FEMININE),
    "ginocchio":("ginocchia",Gender.FEMININE),
    "labbro":   ("labbra",   Gender.FEMININE),
    "osso":     ("ossa",     Gender.FEMININE),
    "uovo":     ("uova",     Gender.FEMININE),
    "dito":     ("dita",     Gender.FEMININE),
    "orecchio": ("orecchie", Gender.FEMININE),
    "muro":     ("muri",     Gender.MASCULINE),   # also "mura" (f.) with different meaning
    "filo":     ("fili",     Gender.MASCULINE),
    "ala":      ("ali",      Gender.FEMININE),
    "arma":     ("armi",     Gender.FEMININE),
    # Feminine irregular
    "moglie":   ("mogli",    Gender.FEMININE),
    # Greco-Latin -ma (masculine)
    "problema": ("problemi", Gender.MASCULINE),
    "tema":     ("temi",     Gender.MASCULINE),
    "sistema":  ("sistemi",  Gender.MASCULINE),
    "programma":("programmi",Gender.MASCULINE),
    "clima":    ("climi",    Gender.MASCULINE),
    "diploma":  ("diplomi",  Gender.MASCULINE),
    "dramma":   ("drammi",   Gender.MASCULINE),
    "poema":    ("poemi",    Gender.MASCULINE),
    "panorama": ("panorami", Gender.MASCULINE),
    "teorema":  ("teoremi",  Gender.MASCULINE),
    "trauma":   ("traumi",   Gender.MASCULINE),
    "schema":   ("schemi",   Gender.MASCULINE),
    "dilemma":  ("dilemmi",  Gender.MASCULINE),
    "fantasma": ("fantasmi", Gender.MASCULINE),
    "telegramma":("telegrammi",Gender.MASCULINE),
}

# ─── Invariable words ───────────────────────────────────────────────────────

INVARIABLE_WORDS: set[str] = {
    # Monosyllables
    "re", "blu", "tè", "gru", "gnu",
    # Foreign words
    "sport", "bar", "film", "gas", "test", "tour", "web", "chat",
    "blog", "rock", "pop", "jazz", "bus", "campus", "virus", "bonus",
    "status", "focus", "corpus", "nexus",
    # Words with final accent
    "città", "virtù", "tribù", "gioventù", "servitù", "volontà",
    "età", "università", "libertà", "verità", "qualità", "quantità",
    "realtà", "felicità", "capacità", "difficoltà", "possibilità",
    "caffè", "tè", "perché", "piè",
    # Abbreviations and acronyms (examples)
    "crisi", "analisi", "ipotesi", "tesi", "sintesi", "diagnosi",
    "parentesi", "enfasi", "simbiosi", "metamorfosi", "nevrosi", "psicosi",
    # -ie invariable
    "specie", "serie", "superficie", "effigie",
}

# ─── Suffixes that produce invariable words ────────────────────────────────

INVARIABLE_SUFFIXES = (
    "ù", "à", "è", "ì", "ò",  # final accent
)

# ─── Pluralization rules ─────────────────────────────────────────────────────

def _apply_rules(word: str, gender: Optional[Gender] = None) -> tuple[str, Gender, str]:
    """
    Apply Italian grammatical rules and return (plural, gender, rule).
    gender: if provided, helps disambiguate (e.g. -a masculine vs feminine)
    """
    w = word.lower()

    # ── Special rules for -cia / -gia ──────────────────────────────────
    if w.endswith("cia") or w.endswith("gia"):
        root = w[:-3]
        if root and root[-1] in "aeiou":
            plural = w[:-1] + "e"
            return plural, Gender.FEMININE, f"-{w[-3:]} (preceding vowel) → -{w[-3:-1]}e"
        else:
            plural = w[:-2] + "e"
            return plural, Gender.FEMININE, f"-{w[-3:]} (preceding consonant) → -ce/-ge"

    # ── Masculine in -co / -go ─────────────────────────────────────────
    MASCULINE_CO_CI = {"amico", "nemico", "greco", "medico", "monaco", "porco",
                       "stomaco", "sindaco", "carico", "incarico", "equivoco"}
    MASCULINE_GO_GI = {"asparago", "profugo", "naufrago"}

    if w.endswith("co") and gender in (None, Gender.MASCULINE):
        if w in MASCULINE_CO_CI:
            return w[:-2] + "ci", Gender.MASCULINE, "-co → -ci (proparoxytone)"
        else:
            return w[:-2] + "chi", Gender.MASCULINE, "-co → -chi"

    if w.endswith("go") and gender in (None, Gender.MASCULINE):
        if w in MASCULINE_GO_GI:
            return w[:-2] + "gi", Gender.MASCULINE, "-go → -gi (proparoxytone)"
        else:
            return w[:-2] + "ghi", Gender.MASCULINE, "-go → -ghi"

    # ── Feminine in -ca / -ga → -che / -ghe ────────────────────────────
    if w.endswith("ca"):
        return w[:-2] + "che", Gender.FEMININE, "-ca → -che"
    if w.endswith("ga"):
        return w[:-2] + "ghe", Gender.FEMININE, "-ga → -ghe"

    # ── -io: if i is tonic → -ii, otherwise → -i ──────────────────────
    TONIC_IO = {"addio", "zio", "mio", "dio", "pio", "rio", "rio", "trio", "duo"}
    if w.endswith("io"):
        if w in TONIC_IO:
            return w[:-1] + "i", Gender.MASCULINE, "-io (tonic i) → -ii"
        else:
            return w[:-2] + "i", Gender.MASCULINE, "-io (atonic i) → -i"

    # ── Base rules ──────────────────────────────────────────────────────
    if w.endswith("a"):
        g = gender if gender else Gender.FEMININE
        if g == Gender.MASCULINE:
            return w[:-1] + "i", Gender.MASCULINE, "-a (masculine) → -i"
        return w[:-1] + "e", Gender.FEMININE, "-a → -e"

    if w.endswith("o"):
        return w[:-1] + "i", Gender.MASCULINE, "-o → -i"

    if w.endswith("e"):
        g = gender if gender else Gender.UNKNOWN
        return w[:-1] + "i", g, "-e → -i"

    return w, Gender.UNKNOWN, "no applicable rule (invariable?)"


# ─── Main function ──────────────────────────────────────────────────────────

def pluralize(word: str, gender: Optional[Gender] = None) -> PluralResult:
    """
    Pluralize an Italian word.

    Args:
        word: the singular word
        gender: Gender.MASCULINE / FEMININE if known (improves accuracy)

    Returns:
        PluralResult with plural, gender, applied rule and exception flag
    """
    w = word.strip().lower()

    # 1. Check exceptions
    if w in EXCEPTIONS:
        pl, gen = EXCEPTIONS[w]
        return PluralResult(
            singular=word,
            plural=pl,
            gender=gen,
            rule="irregular exception",
            exception=True,
        )

    # 2. Check exact invariable words
    if w in INVARIABLE_WORDS:
        return PluralResult(
            singular=word,
            plural=word,
            gender=gender or Gender.INVARIABLE,
            rule="invariable word",
            exception=False,
        )

    # 3. Check invariable suffixes (final accents)
    if any(w.endswith(s) for s in INVARIABLE_SUFFIXES):
        return PluralResult(
            singular=word,
            plural=word,
            gender=gender or Gender.FEMININE,
            rule="invariable due to final accent",
            exception=False,
        )

    # 4. Apply grammatical rules
    plural, calc_gender, rule = _apply_rules(w, gender)
    final_gender = gender if gender else calc_gender

    return PluralResult(
        singular=word,
        plural=plural,
        gender=final_gender,
        rule=rule,
        exception=False,
    )


def pluralize_list(words: list[str]) -> list[PluralResult]:
    """Pluralize a list of words."""
    return [pluralize(w) for w in words]


def pluralize_simple(word: str, gender: Optional[Gender] = None) -> str:
    """Simplified version that returns only the plural string."""
    return pluralize(word, gender).plural


# ─── Demo / Test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        # (word, optional_gender)
        ("casa",       None),
        ("libro",      None),
        ("cane",       None),
        ("amica",      Gender.FEMININE),
        ("amico",      Gender.MASCULINE),
        ("fuoco",      Gender.MASCULINE),
        ("banca",      None),
        ("dialogo",    Gender.MASCULINE),
        ("problema",   None),
        ("sistema",    None),
        ("uomo",       None),
        ("braccio",    None),
        ("dito",       None),
        ("uovo",       None),
        ("moglie",     None),
        ("specie",     None),
        ("città",      None),
        ("caffè",      None),
        ("bar",        None),
        ("film",       None),
        ("crisi",      None),
        ("analisi",    None),
        ("camicia",    None),
        ("arancia",    None),
        ("spiaggia",   None),
        ("frangia",    None),
        ("zio",        None),
        ("negozio",    None),
        ("tema",       None),
        ("gioventù",   None),
    ]

    print("=" * 65)
    print(f"{'SINGULAR':<16} {'PLURAL':<16} {'GENDER':<12} {'RULE'}")
    print("=" * 65)

    for word, gender in test_cases:
        r = pluralize(word, gender)
        flag = " ⚠" if r.exception else ""
        print(
            f"{r.singular:<16} {r.plural:<16} "
            f"{r.gender.value:<12} {r.rule}{flag}"
        )

    print("=" * 65)
    print("\n⚠  = irregular exception\n")
