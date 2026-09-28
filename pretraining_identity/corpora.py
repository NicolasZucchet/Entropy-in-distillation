"""The corpora each cell is measured on, pinned to exact files and revisions.

-- `olmo_mix` and `dolmino`, the OLMo 2 stage-1 and stage-2 pretraining mixes:
   one shard per source (per directory for the stage-2 math sources), tokenized
   and packed into 2048-token blocks. Every source gets the same token budget and
   `plot.py` re-pools the per-source means at the published mixture proportions.
-- `pile`: three 2M-token chunks of Pythia's pre-tokenized, pre-shuffled training
   stream, from the start of shards 0, 10 and 19.
-- `tulu3_sft`, `tulu3_sft_0225`: the SFT mixtures of the OLMo 2 SFT models, read
   with the chat template and supervised on assistant tokens only.
"""
import os

SCRATCH = os.environ.get("SCRATCH_DIR", os.path.join(os.getcwd(), "scratch"))

# (source, file[, share of the source's token budget]) per mix.
OLMO_SHARDS = {
    "olmo_mix": ("allenai/olmo-mix-1124", "99ee6aaace88779d1ef099d36251b91101c1679b", [
        ("dclm", "data/dclm/raw/hero-run-fasttext_for_HF/filtered/OH_eli5_vs_rw_v2_bigram_200k_train/fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data/global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zstd"),
        ("wiki", "data/wiki/wiki-0000.json.gz"),
        ("pes2o", "data/pes2o/pes2o-0000.json.gz"),
        ("arxiv", "data/arxiv/train/arxiv-train-0000.json.gz"),
        ("starcoder", "data/starcoder/v1-decon-100_to_20k-2star-top_token_030/documents/ada-0000.json.gz"),
        ("algebraic-stack", "data/algebraic-stack/train/algebraic-stack-train-0000.json.gz"),
        ("open-web-math", "data/open-web-math/train/open-web-math-train-0000.json.gz"),
    ]),
    "dolmino": ("allenai/dolmino-mix-1124", "a319f19eef1e257417b11ea8c30da266ae175557", [
        ("dclm", "data/dclm/0000/dclm-0000.json.zst"),
        ("wiki", "data/wiki/wiki-0000.json.gz"),
        ("pes2o", "data/pes2o/pes2o-0000.json.gz"),
        ("stackexchange", "data/stackexchange/stackexchange-0000.json.gz"),
        ("flan", "data/flan/tulu_flan-0000.json.gz"),
    ] + [
        # Stage-2 math is seven sources sampled uniformly (dataset card), each a
        # source of its own here. A source spread over several directories reads
        # the first file of each, splitting its budget by their share of its bytes.
        (f"math/{source}", f"data/math/{source}/{path}", share)
        for source, path, share in [
            ("tinyGSM-MIND", "2students/tiny_gsm_inline_part000.000000.jsonl.gz", 0.5142),
            ("tinyGSM-MIND", "problem-solving/tiny_gsm_inline_part000.000000.jsonl.gz", 0.4858),
            ("mathcoder2-synthmath", "ajibawa-2023/Maths-College-0-Final.jsonl", 0.2769),
            ("mathcoder2-synthmath", "m-a-p_Matrix/filtered-math/book_math.0000.0000.jsonl", 0.7231),
            ("tulu_math", "personahub_interm_alg/train_0000.jsonl", 0.0807),
            ("tulu_math", "personahub_math_grade/train_0000.jsonl", 0.1165),
            ("tulu_math", "personahub_math_v5/train_0000.jsonl", 0.8028),
            ("metamath-owmfilter", "0000.jsonl.gz", 1.0),
            ("dolmino_math_synth", "basic_math/basic_math_mj_add_3shot_TRAIN.jsonl", 0.3716),
            ("dolmino_math_synth", "gsm8k-synth/resample_v1_6x/gsm8k_resampled_v1_6x.jsonl", 0.0652),
            ("dolmino_math_synth", "gsm_mind/debate/concat_train.00.jsonl", 0.0645),
            ("dolmino_math_synth", "gsm_mind/interview/concat_train.00.jsonl", 0.0938),
            ("dolmino_math_synth", "gsm_mind/layman-knowall/concat_train.00.jsonl", 0.0861),
            ("dolmino_math_synth", "gsm_mind/probem-solving/concat_train.00.jsonl", 0.0752),
            ("dolmino_math_synth", "gsm_mind/professors/concat_train.00.jsonl", 0.0847),
            ("dolmino_math_synth", "gsm_mind/students/concat_train.00.jsonl", 0.0648),
            ("dolmino_math_synth", "gsm_mind/teacher-student/concat_train.00.jsonl", 0.0941),
            ("gsm8k", "main/train/0.jsonl.zst", 0.4204),
            ("gsm8k", "socratic/train/0.jsonl.zst", 0.4910),
            ("gsm8k", "socratic/test/0.jsonl.zst", 0.0886),
            ("codesearchnet-owmfilter", "go/0000.jsonl.gz", 0.1039),
            ("codesearchnet-owmfilter", "java/0000.jsonl.gz", 0.1432),
            ("codesearchnet-owmfilter", "javascript/0000.jsonl.gz", 0.2450),
            ("codesearchnet-owmfilter", "php/0000.jsonl.gz", 0.3420),
            ("codesearchnet-owmfilter", "python/0000.jsonl.gz", 0.1517),
            ("codesearchnet-owmfilter", "ruby/0000.jsonl.gz", 0.0141),
        ]
    ]),
}

PILE_REPO = "EleutherAI/pile-standard-pythia-preshuffled"
PILE_SEQ = 2049  # the stream is packed into 2049-token sequences
PILE_SHARD_BYTES = 30_000_000_000  # every .bin shard but the last
PILE_CHUNK_TOKENS = 2_000_000
# (source label, shard index, sha256 of the downloaded chunk)
PILE_CHUNKS = [
    ("pile_early", 0, "a81fab9f522cc9c02a382bca36165a848918e90816836740cccd5dc9174cf754"),
    ("pile_mid", 10, "2a4566c6b4108a281111e62f278adfd0262ce6500c9b33791c31fc4f0e6771d4"),
    ("pile_late", 19, "7886c5ca97ebd29a493153679ad60d4f73890683dfe20b91bbb771dbda619cde"),
]

# (dataset repo, commit). 7B and 13B were finetuned on the first, 1B on the second.
SFT_MIXTURES = {
    "tulu3_sft": ("allenai/tulu-3-sft-olmo-2-mixture", "2d335ef93d9ad35e718e201c2aabefa08032a3d3"),
    "tulu3_sft_0225": ("allenai/tulu-3-sft-olmo-2-mixture-0225", "d91a0785ade02942520280fb484866fce41e448f"),
}


def pile_path(label: str) -> str:
    return os.path.join(SCRATCH, "pile", f"{label}.u16")


def shard_paths(mix: str) -> list[tuple[str, str, float]]:
    """`(source, local path, share of the source's budget)` for every shard of a pretraining mix."""
    if mix == "pile":
        return [(label, pile_path(label), 1.0) for label, _, _ in PILE_CHUNKS]
    from huggingface_hub import hf_hub_download

    repo, commit, files = OLMO_SHARDS[mix]
    return [(entry[0], hf_hub_download(repo, entry[1], repo_type="dataset", revision=commit),
             entry[2] if len(entry) > 2 else 1.0)
            for entry in files]
