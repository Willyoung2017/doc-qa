import argparse
from datetime import datetime
from os.path import join
from typing import List, TypeVar, Iterable

from docqa.data_processing.word_vectors import load_word_vectors
import pickle
from docqa.config import SUBSTRING_EMBED_DIR
from os.path import join, exists
import numpy as np


class ResourceLoader(object):
    """
    Abstraction for models the need access to external resources to setup, currently just
    for word-vectors.
    """

    def __init__(self, load_vec_fn=load_word_vectors):
        self.load_vec_fn = load_vec_fn

    def load_word_vec(self, vec_name, voc=None):
        return self.load_vec_fn(vec_name, voc)


class LoadFromPath(object):
    def __init__(self, path):
        self.path = path

    def load_word_vec(self, vec_name, voc=None):
        return load_word_vectors(join(self.path, vec_name), voc, True)


class CachingResourceLoader(ResourceLoader):

    def __init__(self, load_vec_fn=load_word_vectors):
        super().__init__(load_vec_fn)
        self.word_vec = {}

    def load_word_vec(self, vec_name, voc=None):
        if vec_name not in self.word_vec:
            self.word_vec[vec_name] = super().load_word_vec(vec_name)
        return self.word_vec[vec_name]


def print_table(table: List[List[str]]):
    """ Print the lists with evenly spaced columns """

    # print while padding each column to the max column length
    col_lens = [0] * len(table[0])
    for row in table:
        for i,cell in enumerate(row):
            col_lens[i] = max(len(cell), col_lens[i])

    formats = ["{0:<%d}" % x for x in col_lens]
    for row in table:
        print(" ".join(formats[i].format(row[i]) for i in range(len(row))))

T = TypeVar('T')


def transpose_lists(lsts: List[List[T]]) -> List[List[T]]:
    return [list(i) for i in zip(*lsts)]


def max_or_none(a, b):
    if a is None or b is None:
        return None
    return max(a, b)


def flatten_iterable(listoflists: Iterable[Iterable[T]]) -> List[T]:
    return [item for sublist in listoflists for item in sublist]


def split(lst: List[T], n_groups) -> List[List[T]]:
    """ partition `lst` into `n_groups` that are as evenly sized as possible  """
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups


def group(lst: List[T], max_group_size) -> List[List[T]]:
    """ partition `lst` into that the mininal number of groups that as evenly sized
    as possible  and are at most `max_group_size` in size """
    if max_group_size is None:
        return [lst]
    n_groups = (len(lst)+max_group_size-1) // max_group_size
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups


def get_output_name_from_cli():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name', '-n', nargs=1, help='name of output to exmaine')

    args = parser.parse_args()
    if args.name:
        out = join(args.name[0] + "-" + datetime.now().strftime("%m%d-%H%M%S"))
        print("Starting run on: " + out)
    else:
        out = "out/run-" + datetime.now().strftime("%m%d-%H%M%S")
        print("Starting run on: " + out)
    return out


def load_pretrained_substring_embedding(embed_file, vocabs, subdim, init_scale=0.05):
    """

    :param embed_file: name of the embedding file
    :param vocabs: dict of substring vocabs
    :param subdim:
    :return: embed matrix np array of shape (len of vocabs, vocab dim)
    """
    if not exists(embed_file):
        embed_file = join(SUBSTRING_EMBED_DIR, embed_file)
    embed_dict = {}
    with open(embed_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines()[1:]:  # skip number of vectors
            word_ix = line.find(" ")
            word = line[:word_ix]
            if (vocabs is None) or (word.lower() in vocabs):
                embed_dict[word] = np.array([float(x) for x in line[word_ix + 1:-1].split(" ")], dtype=np.float32)

    vocab_size = len(vocabs) + 2  # zero, UNK, ...
    embeddings = 2 * init_scale * np.random.random((vocab_size, subdim)) - init_scale # [-init_scale, init_scale]

    pretrained = 0
    for vocab, idx in vocabs.items():
        if vocab in embed_dict:
            embeddings[idx, :] = embed_dict[vocab]
            pretrained += 1
    print("Loaded {} substring vectors of {} substring".format(pretrained, len(vocabs)))
    return embeddings
