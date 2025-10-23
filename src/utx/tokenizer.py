from dataclasses import dataclass
from typing import List


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
BYTE_OFFSET = 3
VOCAB_SIZE = 256 + BYTE_OFFSET


@dataclass
class ByteTokenizer:
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(BOS_ID)
        for b in text.encode("utf-8"):
            ids.append(int(b) + BYTE_OFFSET)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: List[int]) -> str:
        bytes_list: List[int] = []
        for i in ids:
            if i in (PAD_ID, BOS_ID, EOS_ID):
                continue
            bytes_list.append(i - BYTE_OFFSET)
        return bytes(bytes_list).decode("utf-8", errors="ignore")


