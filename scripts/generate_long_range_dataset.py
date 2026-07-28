"""Generate a long-range agreement dataset from single-token words."""

from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from wordfreq import zipf_frequency

NOUNS = [
    # singular, plural, gender
    ("mother", "mothers", "f"),
    ("sister", "sisters", "f"),
    ("wife", "wives", "f"),
    ("father", "fathers", "m"),
    ("brother", "brothers", "m"),
    ("husband", "husbands", "m"),
]
VERBS = [("knows", "know"), ("helps", "help")]
PREPOSITIONS = ["near", "behind"]
NUMBERS = ["sg", "pl"]
ATTACHMENTS = ["peripheral", "center_embedding"]


def form(word, number):
    return word[0] if number == "sg" else word[1]


def generate():
    rows = []
    for subj, obj, embed in tqdm(list(product(NOUNS, repeat=3))):
        if len({subj[0], obj[0], embed[0]}) < 3:
            continue
        for subj_num, obj_num, embed_num in product(NUMBERS, repeat=3):
            for verb, prep, attachment in product(VERBS, PREPOSITIONS, ATTACHMENTS):
                subj_word = form(subj, subj_num)
                obj_word = form(obj, obj_num)
                embed_word = form(embed, embed_num)
                verb_word = form(verb, subj_num)
                if attachment == "peripheral":
                    words = ["the", subj_word, verb_word, "the", obj_word, prep, "the", embed_word]
                else:
                    words = ["the", subj_word, prep, "the", embed_word, verb_word, "the", obj_word]
                rows.append(
                    {
                        "sentence": " ".join(words) + ".",
                        "prep_LEMMA": prep,
                        "sentence_PP_attached": attachment,
                        "subj_NUM": subj_num,
                        "subj_GEN": subj[2],
                        "subj_ZIPF": zipf_frequency(subj[0], "en"),
                        "obj_NUM": obj_num,
                        "obj_GEN": obj[2],
                        "obj_ZIPF": zipf_frequency(obj[0], "en"),
                        "embedobj_NUM": embed_num,
                        "embedobj_GEN": embed[2],
                        "embedobj_ZIPF": zipf_frequency(embed[0], "en"),
                        "verb_LEMMA": verb[1],
                    }
                )
    df = pd.DataFrame(rows)
    if len(df) != 7680 or not df["sentence"].is_unique:
        raise RuntimeError("Unexpected dataset size or duplicate sentences")
    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("datasets/long_range_agreement_2.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(args.output, index=False)
    print(f"Dataset written to {args.output} ({len(df)} sentences)")
