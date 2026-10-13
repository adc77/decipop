import random

from decision_lab.data.schema import Atom, Rule, Theory


def render_atom(atom: Atom, names: dict[str, str]) -> str:
    arguments = [names[arg] if arg.startswith("?") else arg for arg in atom.arguments]
    if len(arguments) == 1:
        verb = "is not" if atom.negative else "is"
        return f"{arguments[0]} {verb} {atom.predicate}"
    subject, target = arguments
    if atom.negative:
        return f"{subject} does not {atom.predicate.removesuffix('s')} {target}"
    return f"{subject} {atom.predicate} {target}"


def render_rule(rule: Rule, template: str) -> str:
    variables = sorted(
        {arg for atom in rule.conditions for arg in atom.arguments if arg.startswith("?")}
    )
    names = {variable: f"person {index + 1}" for index, variable in enumerate(variables)}
    body = " and ".join(render_atom(atom, names) for atom in rule.conditions)
    head = render_atom(rule.conclusion, names)
    if template == "canonical":
        return f"For every person mentioned: if {body}, then {head}."
    if template == "conditional":
        return f"Whenever {body}, {head}; this applies to all people mentioned."
    if template == "paraphrase":
        return f"For any people mentioned, {head} provided that {body}."
    raise ValueError(f"Unknown rendering template: {template}")


def render(theory: Theory, query: Atom, template: str, rng: random.Random):
    facts = [render_atom(atom, {}) + "." for atom in theory.facts]
    rules = [render_rule(rule, template) for rule in theory.rules]
    rng.shuffle(facts)
    rng.shuffle(rules)
    question = render_atom(query, {}) + "."
    return tuple(facts), tuple(rules), question, build_prompt(facts, rules, question)


def build_prompt(facts, rules, question):
    instruction = (
        "Decide using only these facts and rules. TRUE means the statement follows; "
        "FALSE means its explicit negation follows; UNKNOWN means neither follows. "
        "Absence is not negation. Answer A for TRUE, B for FALSE, or C for UNKNOWN."
    )
    sections = [
        instruction,
        "Facts:\n" + "\n".join(facts),
        "Rules:\n" + "\n".join(rules),
        "Statement: " + question,
        "Answer:",
    ]
    return "\n\n".join(sections)
