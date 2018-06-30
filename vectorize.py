#!/usr/bin/python
# -*- coding: utf-8 -*-

import itertools
import numpy as np
from gensim.models.word2vec import Word2Vec

import core.env as env

from core.source.lexicon import Lexicon
from core.source.opinion import OpinionCollection
from core.source.entity import EntityCollection
from core.source.news import News
from core.source.vectors import OpinionVectorCollection, OpinionVector
from core.source.synonyms import SynonymsCollection

from core.relations import Relation

from classifiers.features.distance import DistanceFeature
from classifiers.features.similarity import SimilarityFeature
from classifiers.features.lexicon import LexiconFeature
from classifiers.features.pattern import PatternFeature
from classifiers.features.entities import EntitiesBetweenFeature
from classifiers.features.entities import EntityTagFeature
from classifiers.features.entities import EntitySemanticClass
from classifiers.features.prepositions import PrepositionsCountFeature
from classifiers.features.frequency import EntitiesFrequency
from classifiers.features.appearance import EntityAppearanceFeature
from classifiers.features.context import ContextFeature

from core.processing.prefix import SentimentPrefixProcessor

import io_utils


def vectorize_opinions(news, entities, opinion_collections):
    """ Vectorize news of train collection that has opinion labeling
    """
    def is_ignored(entity_value):
        ignored = io_utils.get_ignored_entity_values()
        entity_value = env.stemmer.lemmatize_to_str(entity_value)
        if entity_value in ignored:
            # print "ignored: '{}'".format(entity_value.encode('utf-8'))
            return True
        return False

    def get_appropriate_entities(opinion_value, synonyms):
        if synonyms.has_synonym(opinion_value):
            return filter(
                lambda s: entities.has_entity_by_value(s),
                synonyms.get_synonyms_list(opinion_value))
        elif entities.has_entity_by_value(opinion_value):
            return [opinion_value]
        else:
            return []

    collection = OpinionVectorCollection()
    for opinions in opinion_collections:
        for o in opinions:

            left_values = get_appropriate_entities(o.value_left, opinions.synonyms)
            right_values = get_appropriate_entities(o.value_right, opinions.synonyms)

            # TODO. We guarantee that these left and right values are not lemmatized
            if len(left_values) == 0:
                print "Appropriate entity for '{}'->'...' has not been found".format(
                    o.value_left.encode('utf-8'))
                continue

            if len(right_values) == 0:
                print "Appropriate entity for '...'->'{}' has not been found".format(
                   o.value_right.encode('utf-8'))
                continue

            r_count = 0
            relations = []

            for entity_left in left_values:
                for entity_right in right_values:
                    if is_ignored(entity_left):
                        continue

                    if is_ignored(entity_right):
                        continue

                    entities_left_ids = entities.get_entity_by_value(entity_left)
                    entities_right_ids = entities.get_entity_by_value(entity_right)

                    r_count = len(entities_left_ids) * len(entities_right_ids)

                    for e1_ID in entities_left_ids:
                        for e2_ID in entities_right_ids:
                            e1 = entities.get_entity_by_id(e1_ID)
                            e2 = entities.get_entity_by_id(e2_ID)
                            r = Relation(e1.ID, e2.ID, news)
                            relations.append(r)

            if r_count == 0:
                continue

            r_features = np.concatenate(
                [f.calculate(relations) for f in FEATURES], axis=0)

            vector = OpinionVector(o.value_left, o.value_right, r_features, o.sentiment)

            collection.add_vector(vector)

    return collection


def filter_neutral(neutral_opins, limit=10):

    scored_opinions = []
    for o in neutral_opins:

        se_l = 0
        for s in neutral_opins.synonyms.get_synonyms_list(o.value_left):
            if entities.has_entity_by_value(s):
                se_l += 1

        se_r = 0
        for s in neutral_opins.synonyms.get_synonyms_list(o.value_right):
            if entities.has_entity_by_value(s):
                se_r += 1

        popularity = se_l * se_r
        scored_opinions.append((o, popularity))

    scored_opinions.sort(key= lambda x: x[1], reverse=True)
    for o, score in scored_opinions[limit:]:
        neutral_opins.remove_opinion(o)


def save_feature_names(features_filepath):
     features = [f.feature_function_names() for f in FEATURES]
     features = list(itertools.chain.from_iterable(features))
     with open(features_filepath, 'w') as out:
         for f in features:
             out.write('{}\n'.format(f))

#
# Main
#

w2v_model = Word2Vec.load_word2vec_format(io_utils.get_w2v_model_filepath(), binary=True)
prefix_processor = SentimentPrefixProcessor.from_file("data/prefixes.csv")
prepositions_list = io_utils.read_prepositions("data/prepositions.txt")
capitals_list = io_utils.read_lss("data/capitals.lss")
states_list = io_utils.read_lss("data/states.lss")
synonyms = SynonymsCollection.from_file(io_utils.get_synonyms_filepath())
lexicon = Lexicon.from_file("data/rusentilex.csv")

FEATURES = [
    DistanceFeature(),
    SimilarityFeature(w2v_model),
    LexiconFeature(lexicon, prefix_processor),
    PatternFeature([',']),
    EntitiesBetweenFeature(),
    PrepositionsCountFeature(prepositions_list),
    EntitiesFrequency(synonyms),
    EntityAppearanceFeature(),
    EntityTagFeature(),
    EntitySemanticClass(capitals_list, "capitals"),
    EntitySemanticClass(states_list, "states"),
    ContextFeature(lexicon),
]

#
# Train collection
#
root = io_utils.train_root()
for n in io_utils.train_indices():
    entity_filepath = root + "art{}.ann".format(n)
    opin_filepath = root + "art{}.opin.txt".format(n)
    neutral_filepath = root + "art{}.neut.txt".format(n)
    news_filepath = root + "art{}.txt".format(n)
    vector_output = root + "art{}.vectors.txt".format(n)

    print vector_output

    entities = EntityCollection.from_file(entity_filepath)
    news = News.from_file(news_filepath, entities)

    sentiment_opins = OpinionCollection.from_file(
        opin_filepath, io_utils.get_synonyms_filepath())
    neutral_opins = OpinionCollection.from_file(
        neutral_filepath, io_utils.get_synonyms_filepath())

    # filter_neutral(neutral_opins)

    vectors = vectorize_opinions(
        news, entities, [sentiment_opins, neutral_opins])

    vectors.save(vector_output)

#
# Test collection
#
root = io_utils.test_root()
for n in io_utils.test_indices():
    entity_filepath = root + "art{}.ann".format(n)
    news_filepath = root + "art{}.txt".format(n)
    vector_output = root + "art{}.vectors.txt".format(n)
    neutral_filepath = root + "art{}.neut.txt".format(n)

    print vector_output

    entities = EntityCollection.from_file(entity_filepath)
    news = News.from_file(news_filepath, entities)
    neutral_opins = OpinionCollection.from_file(
        neutral_filepath,
        io_utils.get_synonyms_filepath())

    vectors = vectorize_opinions(
        news, entities, [neutral_opins])

    vectors.save(vector_output)

#
# Feature names
#
save_feature_names(io_utils.get_feature_names_filepath())
