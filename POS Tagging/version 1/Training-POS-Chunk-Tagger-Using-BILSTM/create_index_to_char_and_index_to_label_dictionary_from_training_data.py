"""Create index to character and index to pos, chunk label dictionaries and save them as pickle files."""
from argparse import ArgumentParser
from pickle import dump


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def find_unique_characters_and_labels_from_lines(lines):
    """Find unique characters, pos, chunk tags from lines in training file."""
    characters = set()
    pos_tags = set()
    chunk_tags = set()
    for line in lines:
        token, pos, chunk = line.split('\t')
        characters.update(set(token))
        pos_tags.add(pos)
        chunk_tags.add(chunk)
    return characters, pos_tags, chunk_tags


def create_index_to_char_dict(characters):
    """Create a index to dictionary from a set of characters."""
    return {index + 1: char for index, char in enumerate(characters)}


def create_index_to_label_dict(labels):
    """Create index to label dictionary from a set of labels."""
    return {index: label for index, label in enumerate(labels)}


def dump_object_into_pickle_file(data_object, pickle_file):
    """Dump an object into a pickle file."""
    with open(pickle_file, 'wb') as pickle_dump:
        dump(data_object, pickle_dump)


def main():
    """Pass arguments and call functions here."""
    parser = ArgumentParser()
    parser.add_argument('--train', dest='train', help='Enter the training conll file')
    parser.add_argument('--lang', dest='lang', help='Enter the languages usually the 3 digit ISO code')
    args = parser.parse_args()
    train_lines = read_lines_from_file(args.train)
    train_characters, train_pos, train_chunk = find_unique_characters_and_labels_from_lines(train_lines)
    index_to_char_dict = create_index_to_char_dict(train_characters)
    index_to_pos_dict = create_index_to_label_dict(train_pos)
    index_to_chunk_dict = create_index_to_label_dict(train_chunk)
    # Save the index to char dictionary
    dump_object_into_pickle_file(index_to_char_dict, args.lang + '-index-to-char.pkl')
    # Save the index to POS dictionary
    dump_object_into_pickle_file(index_to_pos_dict, args.lang + '-index-to-pos.pkl')
    # Save the index to Chunk dictionary
    dump_object_into_pickle_file(index_to_chunk_dict, args.lang + '-index-to-chunk.pkl')


if __name__ == '__main__':
    main()
