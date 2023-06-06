from typing import List
import re

word_attributes = [
    'word_id',
]

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
    suffixes: str

    # wx format
    #     tha -> d
    #     dha -> x

class Chunk:
    chunk_id: int
    chunk_group: str
    list_of_words: List[Word]

class Sentence:
    id: int
    list_of_chunks = List[Chunk]

    

list_of_sentences = []
s = None
c = None
w = None

with open(r"C:\Users\student\Desktop\200905138\Kannada-NLP\Parsing\Dataset\DL-DL MT\DL-DL MT\Set1_governance_translated_part1_00001-00050.txt", mode='r',  encoding='utf-8') as f:
    for line_number, line in enumerate(f):
        # print(line)
        if line_number < 4:
            continue

        tab_sep = line.strip().split('\t')
        # print(tab_sep)
        if(tab_sep==['']):
            continue

        elif(tab_sep[0]=='))'):
            s.list_of_chunks.append(c)
            continue

        elif(line.startswith("</document>")):
            continue

        elif(line.startswith("</Sentence>")):
            list_of_sentences.append(s)
            continue
        
       
        if(line.startswith("<Sentence")):
            s = Sentence()
            s.list_of_chunks = []
            s.id = re.search(r'\d+', line).group()
            continue

        if(tab_sep[1]=='(('):
            c = Chunk()
            c.list_of_words = []
            c.chunk_id = int(tab_sep[0])
            c.chunk_group = tab_sep[2]
            continue
    
        
    
            
        

        w = Word()

        ########################### HARD CODED - SKIPPING CHUNK 8 --> Line 1727 
        try:
            index_of_dot = tab_sep[0].index('.')
        except:
            continue
        ########################### HARD CODED

        w.word_id = tab_sep[0][index_of_dot + 1]

        w.kannada_word = tab_sep[1]

        w.pos = tab_sep[2]


        fsaf = tab_sep[3]
        index_of_start = fsaf.index("'")
        fsaf = fsaf[index_of_start+1: -2]

        fsaf_words = fsaf.split(',')
        rootword = fsaf_words[0]
        lexical_category = fsaf_words[1]
        gender = fsaf_words[2]
        number = fsaf_words[3]
        person = fsaf_words[4]
        mode = fsaf_words[5]
        suffixes = fsaf_words[6]

        c.list_of_words.append(w)

for sentence in list_of_sentences:
    print(sentence.id)

    for chunk in sentence.list_of_chunks:
        print(chunk.chunk_id)
        print(chunk.chunk_group)
        for word in chunk.list_of_words:
            print(word.kannada_word)

    break