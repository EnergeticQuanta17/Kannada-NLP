import spacy
from pprint import pprint

cls = spacy.util.get_lang_class("kn")
nlp = cls()
nlp.add_pipe("tagger").initialize()

pprint(dir(nlp))
print()
for i in nlp.pipe("ಕನ್ನಡ"):
    print(i)
    
for i in nlp.pipe("ಕನ್ನಡ"):
    print(i)