#!python3


def count_consecutive_consonants(word):
    vowels = "aeiou"
    max_consecutive = 0
    current_consecutive = 0

    for char in word.lower():
        if char.isalpha() and char not in vowels:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    return max_consecutive


def count_consecutive_doubles(word):
    max_consecutive = 0
    current_consecutive = 0

    word = word.lower()
    idx = 0
    while idx < len(word) - 1:
        char = word[idx]
        nextchar = word[idx + 1]

        if char.isalpha() and char == nextchar:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
            idx += 2
        else:
            current_consecutive = 0
            idx += 1

    return max_consecutive


def main():
    results = {}

    with open("dictionary.txt", "r") as file:
        for line in file:
            word = line.strip()
            if word:
                results[word] = count_consecutive_doubles(word)

    return results


if __name__ == "__main__":
    consonant_counts = main()
    for word, count in sorted(consonant_counts.items(), key=lambda kv: kv[1]):
        print(word, count)
