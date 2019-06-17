import random
import re
import string
import requests
import time

from bs4 import BeautifulSoup


def create_strings_from_file(file_names, text_length, num):
    """
        Create all strings by reading lines in specified files

        P.S. If the file does not contains any blank line,
             better to change line 19-27 into 'lines = [l.strip()[0:text_length] for l in f.readlines()]'

        P.P.S The filename should be alphabet or number.
    """
    strings = []
    for filename in file_names:
        with open(filename, 'r', encoding="utf8") as f:
            content = f.readlines()
            for txt in content:
                txt = txt.strip()
                if txt == u"\n":
                    continue
                offset = random.randint(0, 1000)
                text = [txt[i:i + text_length] for i in range(offset, len(txt), text_length)]
                lines = []
                for i in range(len(text)):
                    if len((text[i])) == text_length:
                        lines.append(text[i])
                if len(lines) >= num - len(strings):
                        strings.extend(lines[0:num - len(strings)])
                else:
                        strings.extend(lines)
    if len(strings) == 0:
        raise Exception("No lines could be read in file")
    return strings


def create_strings_from_file_random(file_names, num):
    """
        Same as create_strings_from_file but text_length is random.
    """
    strings = []
    if len(file_names)==0:
        file_names[0] = 'files/16.txt'
    for filename in file_names:
        with open(filename, 'r', encoding="utf8") as f:
            lines = [l.strip(' ').strip()[0:random.randint(8,12)] for l in f.readlines()]
            if len(lines) >= num - len(strings):
                    strings.extend(lines[0:num - len(strings)])
            else:
                    strings.extend(lines)
    if len(strings) == 0:
        raise Exception("No lines could be read in file")
    return strings


def create_strings_from_dict(text_length, num, lang_dict):
    """
        Create all strings by picking X random word in the dictionnary
    """

    dict_len = len(lang_dict)
    strings = []
    for i in range(0, num):
        current_string = ""
        for j in range(0, text_length):
            current_string += lang_dict[random.randrange(dict_len)][0]
        strings.append(current_string)
    return strings


def create_strings_from_wikipedia(minimum_length, count, lang):
    """
        Create all string by randomly picking Wikipedia articles and taking sentences from them.
    """
    sentences = []

    while len(sentences) < count:
        # We fetch a random page
        page = requests.get('https://{}.wikipedia.org/wiki/Special:Random'.format(lang))

        soup = BeautifulSoup(page.text, 'html.parser')

        for script in soup(["script", "style"]):
            script.extract()

        # Only take a certain length
        lines = list(filter(
            lambda s:
                len(s.split(' ')) > minimum_length
                and not "Wikipedia" in s
                and not "wikipedia" in s,
            [
                ' '.join(re.findall(r"[\w']+", s.strip()))[0:200] for s in soup.get_text().splitlines()
            ]
        ))

        # Remove the last lines that talks about contributing
        sentences.extend(lines[0:max([1, len(lines) - 5])])

    return sentences[0:count]


def create_strings_randomly(length, allow_variable, count, let, num, sym, lang):
    """
        Create all strings by randomly sampling from a pool of characters.
    """

    # If none specified, use all three
    if True not in (let, num, sym):
        let, num, sym = True, True, True

    pool = ''
    if let:
        if lang == 'cn':
            pool += ''.join([chr(i) for i in range(19968, 40908)]) # Unicode range of CHK characters
        else:
            pool += string.ascii_letters
    if num:
        pool += "0123456789"
    if sym:
        pool += "!\"#$%&'()*+,-./:;?@[\\]^_`{|}~"

    if lang == 'cn':
        min_seq_len = 1
        max_seq_len = 2
    else:
        min_seq_len = 2
        max_seq_len = 10

    strings = []
    for _ in range(0, count):
        current_string = ""
        for _ in range(0, random.randint(1, length) if allow_variable else length):
            seq_len = random.randint(min_seq_len, max_seq_len)
            current_string += ''.join([random.choice(pool) for _ in range(seq_len)])
            current_string += ' '
        strings.append(current_string[:-1])
    return strings
