#!/usr/bin/env python

# Python version of the evaluation script from CoNLL'00-
# Originates from: https://github.com/spyysalo/conlleval.py
# modified to give overlap in surface forms

# Intentional differences:
# - accept any space as delimiter by default
# - optional file argument (default STDIN)
# - option to set boundary (-b argument)
# - LaTeX output (-l argument) not supported
# - raw tags (-r argument) not supported

"""
This is a from scratch rewrite of conlleval.py from Marek Rei, for the WNUT'17 shared task
that is more picky about ill-formed data.

Two metrics are included: entity performance p/r/f and surface form performance p/r/f

Surface forms compares the set of unique surface forms found in the gold data
with those returned by the system, and scores systems based on how well they
found individual surface forms.

Eric Nichols
"""

from __future__ import print_function
from collections import defaultdict, namedtuple
import fileinput
import sys


def get_sents(lines):
    """
    Args:
        lines (Iterable[str]): the lines

    Yields:
        List[str]: the lines delimited by an empty line
    """
    sent = []
    stripped_lines = (line.strip() for line in lines)
    for line in stripped_lines:
        if line == '':
            yield sent
            sent = []
        else:
            sent.append(line)
    yield sent


Token = namedtuple('Token', 'sent_id word_id word bio tag')
wnut_bio = ('B', 'I', 'O')
wnut_tags = ('corporation', 'creative-work', 'group', 'location', 'person', 'product')


def make_tok(word, bio_tag, sent_id=-1, word_id=-1):
    """
    Args:
        word (str): the surface form of the word
        bio_tag (str): the tag with BIO annotation
        sent_id (int): the sentence ID
        word_id (int): the word ID

    Returns:
        Token

    Raises:
        ValueError
    """
    if bio_tag == 'O':
        bio, tag = 'O', 'O'
    else:
        bio, tag = bio_tag.split('-', 1)
        if bio not in wnut_bio or tag not in wnut_tags:
            raise ValueError('Invalid tag: %s %s %d %d' % (word, bio_tag, sent_id, word_id))
    return Token(sent_id, word_id, word, bio, tag)


def token_to_conll(tok):
    """
    Args:
        tok (Token): 

    Returns:
        str:
    """
    return '%s\t%s' % (tok.word, tok.tag if tok.tag == 'O' else '%s-%s' % (tok.bio, tok.tag))


def line_to_toks(line, sent_id=-1, word_id=-1):
    """
    Args:
        line (str): the input line
        sent_id (int): the current sentence ID
        word_id (int): the current word ID

    Returns:
        Dictionary[str,Token]: the gold and guess tokens stored in a dict with keys for gold and guess

    Raises:
        ValueError
    """
    def make_lbl(i):
        return 'gold' if i == 0 else 'sys_%d' % i

    try:
        fields = line.split('\t')
        word = fields[0]
        return {make_lbl(i): make_tok(word, bio_tag, sent_id, word_id)
                for i, bio_tag in enumerate(fields[1:])}
    except ValueError:
        raise ValueError('Invalid line: %s %d %d' % (line, sent_id, word_id))


def sent_to_toks(sent, sent_id=-1):
    """
    Args:
        sent (Iterator[str]): the lines that comprise a sentence
        sent_id (int): the sentence ID

    Returns:
        Dictionary[str,List[Token]]: the gold and guess tokens for each word in the sentence,
        stored in a dict with keys for gold and guess
    """
    toks = defaultdict(list)
    for word_id, line in enumerate(sent):
        for src, tok in line_to_toks(line, sent_id, word_id).items():
            toks[src].append(tok)
    return toks


Entity = namedtuple('Entity', 'words sent_id word_id_start word_id_stop tag')


def entity_to_tokens(entity):
    """
    Args:
        entity (Entity): 

    Returns:
        List[Token]: 
    """
    def get_bio(_i):
        if entity.tag == 'O':
            return 'O'
        elif _i == 0:
            return 'B'
        else:
            return 'I'

    return [Token(entity.sent_id, entity.word_id_start + i, word, get_bio(i), entity.tag)
            for i, word in enumerate(entity.words)]


def entity_to_conll(entity):
    """
    Args:
        entity (Entity): 

    Returns:
        List[str]: a conll-formatted token tag
    """
    return [token_to_conll(tok) for tok in entity_to_tokens(entity)]


def get_phrases(entities):
    """
    Args:
        entities (Iterable[Entity]): 

    Returns:
        Set[Tuple[str]]
    """
    return {entity.words for entity in entities}


def get_phrases_and_tags(entities):
    """
    Args:
        entities (Iterable[Entity]): 

    Returns:
        Set[Tuple[Tuple[str],str]]:
    """
    return {(entity.words, entity.tag) for entity in entities}


def toks_to_entities(toks):
    """
    Args:
        toks (Iterable[Token]): the tokens in a sentence

    Returns:
        Iterable[Entity]: the corresponding entities in a sentence

    Raises:
        ValueError
    """
    def make_entity(tok):
        return Entity((tok.word, ), tok.sent_id, tok.word_id, tok.word_id+1, tok.tag)

    def extend_entity(entity, tok):
        return Entity(entity.words + (tok.word, ), entity.sent_id, entity.word_id_start, tok.word_id+1, entity.tag)

    def reducer(_entities, tok):
        last = _entities.pop()
        if tok.bio == 'I' and tok.tag == last.tag:
            entity = extend_entity(last, tok)
            _entities.append(entity)
        elif tok.bio == 'B' or (tok.bio == 'O' and tok.tag == 'O'):
            entity = make_entity(tok)
            _entities.extend([last, entity])
        # invalid token sequence tag1 => I-tag2: interpret as tag1 => B-tag2
        elif tok.bio == 'I' and tok.tag != last.tag:
            print('Invalid tag sequence: %s => %s' % (last, tok), file=sys.stderr)
            entity = make_entity(tok)
            _entities.extend([last, entity])
        else:
            raise ValueError('Invalid tag sequence: %s %s' % (last, tok))
        return _entities

    return reduce(reducer, toks[1:], [make_entity(toks[0]), ])


def non_other(entity):
    # type: (Entity) -> bool
    """
    Args:
        entity (Entity): 

    Returns:
        bool
    """
    return entity.tag != 'O'


def filter_entities(entities, p):
    """
    Args:
        entities (Iterable[Entity]): the entities in a sentence
        p (Call[[Entity],bool): the predicate

    Returns:
        List(Entity): the entities filtered by predicate p
    """
    return [entity for entity in entities if p(entity)]


def drop_other_entities(entities):
    """
    Args:
        entities (Iterable[Entity]): 

    Returns:
        Iterator[Entity]
    """
    return filter_entities(entities, non_other)


def doc_to_tokses(lines):
    """
    Args:
        lines (Iterable[str]): the lines in a document

    Returns:
        Dictionary[str,List[List[Tokens]]]: a nested list of list of tokens,
        with one list for each sentence, stored in a dict with keys for gold and guess
    """
    sents = get_sents(lines)
    tokses = defaultdict(list)
    for sent_id, sent in enumerate(sents):
        for src, toks in sent_to_toks(sent, sent_id).items():
            tokses[src].append(toks)
    return tokses


def flatten(nested):
    """
    Args:
        nested (Iterable[Iterable[T]]): a nested iterator

    Returns:
        List[T]: the iterator flattened into a list
    """
    return [x for xs in nested for x in xs]


def doc_to_toks(lines):
    """
    Args:
        lines (Iterator[str]): the lines in a document

    Returns:
        Dictionary[str,List[Tokens]]: a lists of all tokens in the document,
        stored in a dict with keys for gold and guess
    """
    return {src: flatten(nested)
            for src, nested in doc_to_tokses(lines).items()}


def doc_to_entitieses(lines):
    """
    Args:
        lines (Iterator[str]): the lines in a document

    Returns:
        Dictionary[str,List[List[Entity]]]: a nested list of lists of entities,
        stored in a dict with keys for gold and guess

    """
    entitieses = defaultdict(list)
    for src, tokses in doc_to_tokses(lines).items():
        entitieses[src] = [toks_to_entities(toks) for toks in tokses]
    return entitieses


def doc_to_entities(lines):
    """
    Args:
        lines (Iterator[str]): the lines in a document

    Returns:
        Dictionary[str,List[Entities]]: a lists of all entities in the document,
        stored in a dict with keys for gold and guess
    """
    return {src: flatten(nested)
            for src, nested in doc_to_entitieses(lines).items()}


def get_tags(entities):
    """
    Args:
        entities (Iterable[Entity]): the entities in a sentence

    Returns:
        Set[str]: a set of their tags, excluding 'O'
    """
    return {entity.tag for entity in entities} - {'O'}


Results = namedtuple('Results', 'gold guess correct p r f')


def get_tagged_entities(entities):
    """
    Args:
        entities (Dict[str,List[Entity]]): 

    Returns:
        Dict[str,List[Entity]]
    """
    return {src: drop_other_entities(entities)
            for src, entities in entities.items()}


def get_correct(gold, guess):
    """
    Args:
        gold (Iterable[T]): 
        guess (Iterable[T]): 

    Returns:
        Set[T]
    """
    return set(gold) & set(guess)


def get_tp(gold, guess):
    """
    Args:
        gold (Iterable[T]): 
        guess (Iterable[T]): 

    Returns:
        Set[T]
    """
    return get_correct(gold, guess)


def get_fn(gold, guess):
    """
    Args:
        gold (Iterable[T]): 
        guess (Iterable[T]): 

    Returns:
        Set[T]
    """
    return set(gold) - set(guess)


def get_fp(gold, guess):
    """
    Args:
        gold (Iterable[T]): 
        guess (Iterable[T]): 

    Returns:
        Set[T]
    """
    return set(guess) - set(gold)


def get_tn(tp, fp, fn, _all):
    """
    Args:
        tp (Set[T]): 
        fp (Set[T]): 
        fn (Set[T]):
        _all (Iterable[T]):

    Returns:
        Set[T]
    """
    return set(_all) - tp - fp - fn


def get_tp_fp_fn_tn(gold, guess, _all):
    """
    Args:
        gold (Iterator[T]): 
        guess (Iterator[T]): 
        _all (Iterator[T]):

    Returns:
        Tuple[Set[str],Set[str],Set[str],Set[str]]:
    """
    tp = get_tp(gold, guess)
    fp = get_fp(gold, guess)
    fn = get_fn(gold, guess)
    tn = get_tn(tp, fp, fn, _all)
    return tp, fp, fn, tn


def get_tp_fp_fn_tn_phrases(gold, guess, _all):
    """
    Args:
        gold: List[Entity]
        guess: List[Entity]
        _all: List[Entity]

    Returns:
        Tuple[Set[str],Set[str],Set[str],Set[str]]:
    """
    all_phrases = get_phrases(_all)
    gold_phrases = get_phrases(gold)
    guess_phrases = get_phrases(guess)
    correct_phrases = get_phrases(get_correct(gold, guess))
    tp = correct_phrases
    fp = guess_phrases - tp
    fn = gold_phrases - tp
    tn = get_tn(tp, fp, fn, all_phrases)
    return tp, fp, fn, tn


def calc_results(gold_entities, guess_entities, surface_form=False):
    """
    Args:
        gold_entities (List[Entity]): the gold standard entity annotations
        guess_entities (List[Entity]): a system's entity guesses
        surface_form (bool): whether or not to calculate f1-scores on the entity surface forms

    Returns:
        Results: the results stored in a namedtuple
    """
    # get the correct system guesses by taking the intersection of gold and guess entities,
    # taking into account tags and document locations
    correct_entities = get_correct(gold_entities, guess_entities)
    if surface_form:  # count only unique surface forms when True
        correct_entities = get_phrases_and_tags(correct_entities)
        gold_entities = get_phrases_and_tags(gold_entities)
        guess_entities = get_phrases_and_tags(guess_entities)

    gold = len(gold_entities)
    guess = len(guess_entities)
    correct = len(correct_entities)

    try:
        p = correct / float(guess)
    except ZeroDivisionError:
        p = 0.0
    try:
        r = correct / float(gold)
    except ZeroDivisionError:
        r = 0.0
    try:
        f = 2.0 * p * r / (p + r)
    except ZeroDivisionError:
        f = 0.0

    return Results(gold, guess, correct, p, r, f)


# noinspection PyListCreation,PyDictCreation
def fmt_results(tokens, all_entities, surface_form=False):
    """
    Args:
        tokens (Dict[str,List[Tokens]): a dictionary of gold and guess tokens
        all_entities (Dict[str,List[Entity]): a dictionary of gold and guess entities
        surface_form (bool): whether or not to calculate f1-scores on the entity surface forms

    Yield:
        str: (near) W-NUT format evaluation results
    """
    _sys = 'sys_1'
    # throw out 'O' tags to get overall p/r/f
    tagged_entities = get_tagged_entities(all_entities)
    results = {'all': calc_results(all_entities['gold'], all_entities[_sys], surface_form=False),
               'tagged': calc_results(tagged_entities['gold'], tagged_entities[_sys], surface_form),
               'tokens': calc_results(tokens['gold'], tokens[_sys], surface_form=False)}

    yield('processed %d tokens with %d phrases; ' %
          (results['tokens'].gold, results['tagged'].gold))
    yield('found: %d phrases; correct: %d.\n' %
          (results['tagged'].guess, results['tagged'].correct))

    if results['tokens'].gold > 0:
        # only use token counts for accuracy
        yield('accuracy: %6.2f%%; ' %
              (100. * results['tokens'].correct / results['tokens'].gold))
        yield('precision: %6.2f%%; ' % (100. * results['tagged'].p))
        yield('recall: %6.2f%%; ' % (100. * results['tagged'].r))
        yield('FB1: %6.2f\n' % (100. * results['tagged'].f))

    # get results for each entity category
    tags = get_tags(all_entities['gold'])
    for tag in sorted(tags):
        entities = {src: filter_entities(entities, lambda e: e.tag == tag)
                    for src, entities in all_entities.items()}
        results = calc_results(entities['gold'], entities[_sys], surface_form)
        yield('%17s: ' % tag)
        yield('precision: %6.2f%%; ' % (100. * results.p))
        yield('recall: %6.2f%%; ' % (100. * results.r))
        yield('FB1: %6.2f  %d\n' % (100. * results.f, results.correct))


def main():
    # get tokens and entities
    lines = [line for line in fileinput.input()]
    tokens = doc_to_toks(lines)
    entities = doc_to_entities(lines)

    # report results
    print("### ENTITY F1-SCORES ###")
    for line in fmt_results(tokens, entities, surface_form=False):
        print(line)
    print()
    print("### SURFACE FORM F1-SCORES ###")
    for line in fmt_results(tokens, entities, surface_form=True):
        print(line)


if __name__ == '__main__':
    main()
