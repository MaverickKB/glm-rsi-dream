"""Local deterministic fact grading with grouped aliases and leakage rules."""
import re
import unicodedata


def normalized(text):
    return unicodedata.normalize("NFKC", text).casefold()


def states_alias(answer, aliases):
    text = normalized(answer)
    for alias in aliases:
        value = normalized(alias).strip()
        if not value:
            raise ValueError("empty fact alias")
        pattern = r"(?<!\w)" + re.escape(value) + r"(?!\w)"
        for match in re.finditer(pattern, text):
            prefix = text[max(0, match.start() - 48):match.start()]
            if re.search(r"\b(?:not|never|isn't|is not|isnt|isn’t|wasn't|wasn’t|unknown|cannot confirm)\b[^.!?;]*$", prefix):
                continue
            return True
    return False


def grade(answer, task):
    if not isinstance(answer, str):
        return False
    text = normalized(answer)
    uncertainty = r"(?:don.t know|do not know|don.t have|do not have|can.t confirm|cannot confirm|not sure|unknown|no information|haven.t|have not|if .*?(?:named|called)|could be called)"
    if re.search(uncertainty, text):
        return False
    aliases = task["aliases"]
    if not aliases:
        raise ValueError("fact needs aliases")
    if isinstance(aliases[0], str):
        return states_alias(answer, aliases)
    for alternatives in aliases:
        if not states_alias(answer, alternatives):
            return False
    return True


def false_trigger(answer, facts):
    for fact in facts:
        rules = fact.get("false_trigger")
        if rules is None:
            if grade(answer, fact):
                return True
            continue
        if states_alias(answer, rules.get("unique_aliases", [])):
            return True
        for pattern in rules.get("association_patterns", []):
            if re.search(pattern, normalized(answer)):
                return True
    return False
