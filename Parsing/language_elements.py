from typing import List


class Word:
    word_id: float
    kannada_word: str
    pos: str

    rootword: str
    lexical_category: str
    gender: str
    number: str
    person: str
    mode: str
    k_suffix: str
    e_suffix: str

    # wx format
    #     tha -> d
    #     dha -> x

class Chunk:
    chunk_id: int
    chunk_group: str
    list_of_words: List[Word]

class Sentence:
    id: int
    list_of_chunks: List[Chunk]