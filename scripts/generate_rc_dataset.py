"""Generate a relative clause (RC) dataset as CSV.

Combinatorially crosses 8 sentence structures (subj/obj × who/that ×
peripheral/central) × noun triples × number variants.
Mixing "who" and "that" relativizers breaks word-position↔clause-type
correlations that are otherwise structurally unavoidable.
"""

from itertools import product

import pandas as pd
from tqdm.auto import tqdm
from wordfreq import zipf_frequency

# ── Lexicon ──────────────────────────────────────────────────────────────────

NOUNS = [
    # (singular, plural, gender)
    ("woman", "women", "f"),
    ("girl", "girls", "f"),
    ("queen", "queens", "f"),
    ("lady", "ladies", "f"),
    ("man", "men", "m"),
    ("boy", "boys", "m"),
    ("king", "kings", "m"),
]

VERBS = [
    # (singular, plural)
    ("sees", "see"),
    ("likes", "like"),
]

CONNECTORS = [
    "then",
    "after",
    "before",
]

# ── Helpers ──────────────────────────────────────────────────────────────────


def noun_form(noun, num):
    """Return the surface form for a noun given number ('sg' or 'pl')."""
    sg, pl, _ = noun
    return sg if num == "sg" else pl


def verb_form(verb, num):
    """Return the surface form for a verb given subject number."""
    sg, pl = verb
    return sg if num == "sg" else pl


def build_sentence(
    subj_n,
    obj_n,
    embed_n,
    subj_num,
    obj_num,
    embed_num,
    main_verb,
    rc_verb,
    rc_type,
    rel,
    attachment,
):
    """Build the sentence string and determine verb agreement."""
    subj = noun_form(subj_n, subj_num)
    obj_ = noun_form(obj_n, obj_num)
    embed = noun_form(embed_n, embed_num)

    # Main verb agrees with main-clause subject
    mv = verb_form(main_verb, subj_num)

    # RC verb agreement
    if rc_type == "subj":
        # RC subject = noun the RC modifies
        rc_subj_num = obj_num if attachment == "peripheral" else subj_num
    else:  # obj
        rc_subj_num = embed_num
    rv = verb_form(rc_verb, rc_subj_num)

    # Build word list
    if attachment == "peripheral":
        if rc_type == "subj":
            words = ["the", subj, mv, "the", obj_, rel, rv, "the", embed]
        else:
            words = ["the", subj, mv, "the", obj_, rel, "the", embed, rv]
    else:  # central
        if rc_type == "subj":
            words = ["the", subj, rel, rv, "the", embed, mv, "the", obj_]
        else:
            words = ["the", subj, rel, "the", embed, rv, mv, "the", obj_]

    sentence = " ".join(words) + "."
    return sentence, words


def build_non_rc_sentence(
    subj_n,
    obj_n,
    embed_n,
    subj_num,
    obj_num,
    embed_num,
    verb1,
    verb2,
    connector,
    template,
):
    """Build a simple 2-clause sentence without relative clause.

    Template A: the SUBJ VERB the OBJ CONNECTOR the EMBED VERB  (connector at word_6)
    Template B: the SUBJ CONNECTOR the OBJ VERB the EMBED VERB  (connector at word_3)
    """
    subj = noun_form(subj_n, subj_num)
    obj_ = noun_form(obj_n, obj_num)
    embed = noun_form(embed_n, embed_num)

    v1 = verb_form(verb1, subj_num)
    v2 = verb_form(verb2, embed_num)

    if template == "A":
        # the SUBJ VERB the OBJ CONNECTOR the EMBED VERB
        words = ["the", subj, v1, "the", obj_, connector, "the", embed, v2]
    else:
        # the SUBJ CONNECTOR the OBJ VERB the EMBED VERB
        words = ["the", subj, connector, "the", obj_, v1, "the", embed, v2]

    sentence = " ".join(words) + "."
    return sentence, words


# ── Main ─────────────────────────────────────────────────────────────────────

RC_TYPES = ["subj", "obj"]
RELATIVIZERS = ["who", "that"]
ATTACHMENTS = ["peripheral", "central"]
NUMBERS = ["sg", "pl"]

rows = []
for subj_n, obj_n, embed_n in tqdm(list(product(NOUNS, repeat=3))):
    # Require all three lemmas to be distinct
    if subj_n[0] == obj_n[0] or subj_n[0] == embed_n[0] or obj_n[0] == embed_n[0]:
        continue
    for subj_num, obj_num, embed_num in product(NUMBERS, repeat=3):
        for main_verb, rc_verb in product(VERBS, repeat=2):
            # Skip sentences with the same verb twice
            if main_verb == rc_verb:
                continue
            for rc_type, rel, attachment in product(
                RC_TYPES, RELATIVIZERS, ATTACHMENTS
            ):
                _, words = build_sentence(
                    subj_n,
                    obj_n,
                    embed_n,
                    subj_num,
                    obj_num,
                    embed_num,
                    main_verb,
                    rc_verb,
                    rc_type,
                    rel,
                    attachment,
                )
                sentence = " ".join(words) + "."
                rows.append(
                    {
                        "sentence": sentence,
                        "sentence_CLAUSE": rc_type,
                        "sentence_RC_attached": attachment,
                        "subj_NUM": subj_num,
                        "subj_GEN": subj_n[2],
                        "subj_ZIPF": zipf_frequency(subj_n[0], "en"),
                        "obj_NUM": obj_num,
                        "obj_GEN": obj_n[2],
                        "obj_ZIPF": zipf_frequency(obj_n[0], "en"),
                        "embed_NUM": embed_num,
                        "embed_GEN": embed_n[2],
                        "embed_ZIPF": zipf_frequency(embed_n[0], "en"),
                        "verb_ZIPF": zipf_frequency(main_verb[1], "en"),
                        **{f"word_{i}": w for i, w in enumerate(words[1:], start=2)},
                    }
                )

# Generate non-RC sentences
for subj_n, obj_n, embed_n in tqdm(
    list(product(NOUNS, repeat=3)), desc="Non-RC sentences"
):
    # Require all three lemmas to be distinct
    if subj_n[0] == obj_n[0] or subj_n[0] == embed_n[0] or obj_n[0] == embed_n[0]:
        continue
    for subj_num, obj_num, embed_num in product(NUMBERS, repeat=3):
        for verb1, verb2 in product(VERBS, repeat=2):
            # Skip sentences with the same verb twice
            if verb1 == verb2:
                continue
            for connector in CONNECTORS:
                template = "A"
                _, words = build_non_rc_sentence(
                    subj_n,
                    obj_n,
                    embed_n,
                    subj_num,
                    obj_num,
                    embed_num,
                    verb1,
                    verb2,
                    connector,
                    template,
                )
                sentence = " ".join(words) + "."
                rows.append(
                    {
                        "sentence": sentence,
                        "sentence_CLAUSE": pd.NA,
                        "sentence_RC_attached": pd.NA,
                        "subj_NUM": subj_num,
                        "subj_GEN": subj_n[2],
                        "subj_ZIPF": zipf_frequency(subj_n[0], "en"),
                        "obj_NUM": obj_num,
                        "obj_GEN": obj_n[2],
                        "obj_ZIPF": zipf_frequency(obj_n[0], "en"),
                        "embed_NUM": embed_num,
                        "embed_GEN": embed_n[2],
                        "embed_ZIPF": zipf_frequency(embed_n[0], "en"),
                        "verb_ZIPF": zipf_frequency(verb1[1], "en"),
                        **{f"word_{i}": w for i, w in enumerate(words[1:], start=2)},
                    }
                )

df = pd.DataFrame(rows)
df.to_csv("datasets/relative_clause_2.csv", index=False)
print("Dataset written to datasets/relative_clause_2.csv")
