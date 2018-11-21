import argparse
import json
import urllib
from os import listdir, mkdir
from os.path import expanduser, join, exists, dirname
from typing import List

from tqdm import tqdm

from docqa import config
from docqa.squad.squad_data import Question, Document, Paragraph, SquadCorpus
from docqa.data_processing.span_data import ParagraphSpan, ParagraphSpans
from docqa.data_processing.text_utils import get_word_span, space_re, NltkAndPunctTokenizer
from docqa.utils import flatten_iterable

"""
Script to build a corpus from SQUAD training data 
"""

tag_list = []


def clean_title(title):
    """ Squad titles use URL escape formatting, this method undoes it to get the wiki-title"""
    return urllib.parse.unquote(title).replace("_", " ")


def parse_squad_data(source, srl_tag_context_file, srl_tag_question_file, name, tokenizer, use_tqdm=True) -> List[Document]:
    with open(source, 'r') as f:
        source_data = json.load(f)
    tag_context_file = list(open(srl_tag_context_file, 'r'))
    tag_question_file = list(open(srl_tag_question_file, 'r'))

    if use_tqdm:
        iter_files = tqdm(source_data['data'], ncols=80)
    else:
        iter_files = source_data['data']

    for article_ix, article in enumerate(iter_files):
        tag_context = json.loads(tag_context_file[article_ix])
        tag_question = json.loads(tag_question_file[article_ix])
        article_ix = "%s-%d" % (name, article_ix)
        paragraphs = []

        for para_ix, para in enumerate(article['paragraphs']):
            tag_context_para = tag_context['paragraphs'][para_ix]
            tag_question_para =  tag_question['paragraphs'][para_ix]
            questions = []
            context = para['context']

            tokenized = tokenizer.tokenize_with_inverse(context)
            # list of sentences + mapping from words -> original text index
            text, text_spans = tokenized.text, tokenized.spans
            flat_text = flatten_iterable(text)

            n_words = sum(len(sentence) for sentence in text)

            text_tag = []
            for senix, sen in enumerate(tag_context_para['sentences']):
                sen_srl = sen['srl']
                cnt_sen_tag = 0
                tag_ix = 0
                sen_words = sen_srl['words']
                sen_verbs = sen_srl['verbs']
                if len(sen_verbs) == 0:
                    sen_tag = ["O"] * len(sen_words)
                else:
                    for ix, verb_tag in enumerate(sen_verbs):
                        cnt_tmp_tag = 0
                        for tag in verb_tag['tags']:
                            if not tag_list.__contains__(tag):
                                tag_list.append(tag)
                            if tag != 'O':
                                cnt_tmp_tag = cnt_tmp_tag + 1
                        if cnt_tmp_tag > cnt_sen_tag:
                            cnt_sen_tag = cnt_tmp_tag
                            tag_ix = ix
                    sen_tag = sen_verbs[tag_ix]['tags']
                text_tag.append(sen_tag)

            for question_ix, question in enumerate(para['qas']):
                # There are actually some multi-sentence questions, so we should have used
                # tokenizer.tokenize_paragraph_flat here which would have produced slighy better
                # results in a few cases. However all the results we report were
                # done using `tokenize_sentence` so I am just going to leave this way
                question_tags = tag_question_para['qas'][question_ix]
                question_srl = question_tags['srl']
                cnt_question_tag = 0
                tag_ix = 0
                question_words = question_srl['words']
                question_verbs = question_srl['verbs']
                if len(question_verbs) == 0:
                    question_tag = ["O"] * len(question_words)
                else:
                    for ix, verb_tag in enumerate(question_verbs):
                        cnt_tmp_tag = 0
                        for tag in verb_tag['tags']:
                            if not tag_list.__contains__(tag):
                                tag_list.append(tag)
                            if tag != 'O':
                                cnt_tmp_tag = cnt_tmp_tag + 1
                        if cnt_tmp_tag > cnt_question_tag:
                            cnt_question_tag = cnt_tmp_tag
                            tag_ix = ix
                    question_tag = question_verbs[tag_ix]['tags']

                question_text = tokenizer.tokenize_sentence(question['question'])
                answer_spans = []
                for answer_ix, answer in enumerate(question['answers']):
                    answer_raw = answer['text']

                    answer_start = answer['answer_start']
                    answer_stop = answer_start + len(answer_raw)

                    word_ixs = get_word_span(text_spans, answer_start, answer_stop)

                    first_word = flat_text[word_ixs[0]]
                    first_word_span = text_spans[word_ixs[0]]
                    last_word = flat_text[word_ixs[-1]]
                    last_word_span = text_spans[word_ixs[-1]]

                    char_start = answer_start - first_word_span[0]
                    char_end = answer_stop - last_word_span[0]

                    # Sanity check to ensure we can rebuild the answer using the word and char indices
                    # Since we might not be able to "undo" the tokenizing exactly we might not be able to exactly
                    # rebuild 'answer_raw', so just we check that we can rebuild the answer minus spaces
                    if len(word_ixs) == 1:
                        if first_word[char_start:char_end] != answer_raw:
                            raise ValueError()
                    else:
                        rebuild = first_word[char_start:]
                        for word_ix in word_ixs[1:-1]:
                            rebuild += flat_text[word_ix]
                        rebuild += last_word[:char_end]
                        if rebuild != space_re.sub("", tokenizer.clean_text(answer_raw)):
                            raise ValueError(rebuild + " " + answer_raw)

                    # Find the sentence with in-sentence offset
                    sent_start, sent_end, word_start, word_end = None, None, None, None
                    on_word = 0
                    for sent_ix, sent in enumerate(text):
                        next_word = on_word + len(sent)
                        if on_word <= word_ixs[0] < next_word:
                            sent_start = sent_ix
                            word_start = word_ixs[0] - on_word
                        if on_word <= word_ixs[-1] < next_word:
                            sent_end = sent_ix
                            word_end = word_ixs[-1] - on_word
                            break
                        on_word = next_word

                    # Sanity check these as well
                    if text[sent_start][word_start] != flat_text[word_ixs[0]]:
                        raise RuntimeError()
                    if text[sent_end][word_end] != flat_text[word_ixs[-1]]:
                        raise RuntimeError()

                    span = ParagraphSpan(
                        sent_start, word_start, char_start,
                        sent_end, word_end, char_end,
                        word_ixs[0], word_ixs[-1],
                        answer_raw)
                    if span.para_word_end >= n_words or \
                            span.para_word_start >= n_words:
                        raise RuntimeError()
                    answer_spans.append(span)

                questions.append(Question(question['id'], question_text, ParagraphSpans(answer_spans), question_tag))

            paragraphs.append(Paragraph(text, questions, article_ix, para_ix, context, text_spans, text_tag))

        yield Document(article_ix, article["title"], paragraphs)


def main():
    parser = argparse.ArgumentParser("Preprocess SQuAD data")
    parser.add_argument("--train_file", default=config.SQUAD_TRAIN)  # default is the path of squad_train
    parser.add_argument("--dev_file", default=config.SQUAD_DEV)  # default is the path of squad_dev
    parser.add_argument("--tag_file", default=join(config.SQUAD_SOURCE_DIR, "srl_tag"))

    if not exists(config.CORPUS_DIR):
        mkdir(config.CORPUS_DIR)

    args = parser.parse_args()
    train_tag_file = join(args.tag_file, "srl_squad_train")
    train_question_tag_file = join(args.tag_file, "srl_squad_question_train")
    dev_tag_file = join(args.tag_file, "srl_squad_dev")
    dev_question_tag_file = join(args.tag_file, "srl_squad_question_dev")

    tokenzier = NltkAndPunctTokenizer()

    print("Parsing train...")
    train = list(parse_squad_data(args.train_file, train_tag_file, train_question_tag_file, "train", tokenzier))

    print("Parsing dev...")
    dev = list(parse_squad_data(args.dev_file, dev_tag_file, dev_question_tag_file, "dev", tokenzier))

    print("Saving...")
    SquadCorpus.make_corpus(train, dev, tag_list)

    print("Done")


if __name__ == "__main__":
    main()
