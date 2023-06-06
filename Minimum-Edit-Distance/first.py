import unicodedata

def is_valid_unicode(character):
    try:
        unicodedata.name(character)
        return True
    except ValueError:
        return False

kannada_letters = [chr(codepoint) for codepoint in range(0x0C80, 0x0CFF + 1) if is_valid_unicode(chr(codepoint))]
print(kannada_letters)

with open('output.txt', 'w') as f:
    for i, j in enumerate(kannada_letters):
        print(ord(j), j)
