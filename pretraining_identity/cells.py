"""Every (checkpoint, corpus) cell behind the two figures.

`revision` is the branch name on the Hugging Face Hub and `commit` the commit it
resolved to, which is what gets loaded.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cell:
    model: str
    revision: str
    commit: str
    stage: str  # stage1 | base (the released model) | sft | final (Pythia)
    mix: str  # olmo_mix | dolmino | pile | tulu3_sft | tulu3_sft_0225
    # The OLMo 2 base tokenizers have no chat template, so a base model measured
    # on an SFT mixture uses its SFT model's tokenizer, which is otherwise identical.
    tokenizer: tuple[str, str] | None = None
    dtype: str = "bfloat16"
    batch_size: int = 4
    seq_len: int = 2048
    gpus: int = 1

    @property
    def size(self) -> str:
        parts = self.model.split("/")[-1].split("-")
        while parts[-1] == "SFT":
            parts.pop()
        return parts[-1]

    @property
    def name(self) -> str:
        return f"{self.size}-{self.stage}-on-{self.mix}"


OLMO_1B, OLMO_7B, OLMO_13B, OLMO_32B = (
    "allenai/OLMo-2-0425-1B", "allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-13B",
    "allenai/OLMo-2-0325-32B")

# The final pretrained models (`main`, a weight average of several stage-2 runs
# for 7B, 13B and 32B), which the SFT models were finetuned from.
MAIN = {OLMO_1B: "a1847dff35000b4271fa70afc5db10fd29fedbdf",
        OLMO_7B: "7df9a82518afdecae4e8c026b27adccc8c1f0032",
        OLMO_13B: "3fefddc1bf18a30e1d9b91000271630718f2aa8b",
        OLMO_32B: "cc9d3cf9c7230b86ee6b84607b37db1c01e3f1ed"}

# Figure 9: OLMo 2 at the end of stage 1 on the stage-1 mix and the final models
# on the stage-2 mix, and Pythia on its own training stream.
PRETRAINING = [
    Cell(OLMO_1B, "stage1-step1907359-tokens4001B", "9d3e43659f00c17e6da23cf32333afd1fc39fa1a", "stage1", "olmo_mix"),
    Cell(OLMO_7B, "stage1-step928646-tokens3896B", "c0371f4281bf2376207646c6b62ddc6c442c7577", "stage1", "olmo_mix"),
    Cell(OLMO_13B, "stage1-step596057-tokens5001B", "08d2aca2e28ab67ad859793f76ef5c923e94ac11", "stage1", "olmo_mix"),
    Cell(OLMO_32B, "stage1-step721901-tokens6056B", "cd8442669fb9ece49be20a6190b1499fe86dc332", "stage1", "olmo_mix", batch_size=2, gpus=2),
] + [
    Cell(model, "main", commit, "base", "dolmino",
         **({"batch_size": 2, "gpus": 2} if model == OLMO_32B else {}))
    for model, commit in MAIN.items()
] + [
    # Pythia was trained in float16 and loses accuracy in bfloat16.
    Cell(f"EleutherAI/pythia-{size}", "step143000", commit, "final", "pile", dtype="float16", seq_len=2049)
    for size, commit in [
        ("70m", "de3e4e2d6cbb3b1a51f90fe153ea51fcd8c7c852"),
        ("160m", "b56d9bee36300031aeea723b73c4d62ac7fa71a2"),
        ("410m", "bba6a464f54bbf08fc174cfb351d9794d58af21d"),
        ("1b", "c0fea4b917615c60370c0aeb76877f18fef30ab3"),
        ("1.4b", "9cc5c8c8148a4e0115d9e29c6b4f21124cfe748a"),
        ("2.8b", "dbe7ae300a54abcdc475a33907b3dff81d25709f"),
        ("6.9b", "21bfa02e806e253fe453702c29c81d9f83617255"),
        ("12b", "9c5f35826eb1a9558fb4414c033f58e30322a092"),
    ]
]

# Figure 4: each base model (`main`, the checkpoint SFT started from) and its SFT
# model, on the stage-2 pretraining mix and on the SFT mixture.
# (base repo, base commit, SFT repo, SFT commit, SFT mixture)
_SFT_PAIRS = [
    (OLMO_1B, MAIN[OLMO_1B],
     "allenai/OLMo-2-0425-1B-SFT", "0d85a3d037876ce6ac7d4311d994400fc66ac27f", "tulu3_sft_0225"),
    (OLMO_7B, MAIN[OLMO_7B],
     "allenai/OLMo-2-1124-7B-SFT", "1de02c0175118a9de5854aec80a1f970e701e928", "tulu3_sft"),
    (OLMO_13B, MAIN[OLMO_13B],
     "allenai/OLMo-2-1124-13B-SFT", "b7ec47ed94f3f1d244d9d3cc53eb5d269f8eceec", "tulu3_sft"),
]
SFT = [
    cell
    for base, base_commit, sft, sft_commit, mixture in _SFT_PAIRS
    for cell in (
        Cell(base, "main", base_commit, "base", "dolmino"),
        Cell(base, "main", base_commit, "base", mixture, tokenizer=(sft, sft_commit)),
        Cell(sft, "main", sft_commit, "sft", "dolmino"),
        Cell(sft, "main", sft_commit, "sft", mixture),
    )
]

CELLS = {cell.name: cell for cell in PRETRAINING + SFT}
FIGURES = {"families": [c.name for c in PRETRAINING], "sft": [c.name for c in SFT]}
