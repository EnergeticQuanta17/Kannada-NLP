import random

def generate_unique_colors(num_colors):
    unique_colors = set()
    while len(unique_colors) < num_colors:
        unique_colors.add("#%06x" % random.randint(0, 0xFFFFFF))
    return list(unique_colors)

colors = generate_unique_colors(80)
print(colors)
