import numpy as np

from core.relations import Relation
from base import Base


class DistanceFeature(Base):

    def __init__(self):
        pass

    def create(self, relation):
        """ distance in chars between entities of relation
        """
        assert(isinstance(relation, Relation))
        e1 = relation.news.entities.get_entity_by_id(relation.entity_left_ID)
        e2 = relation.news.entities.get_entity_by_id(relation.entity_right_ID)
        lemmas = relation.news.Processed.get_text_between_entities_to_lemmatized_list(e1, e2)
        return np.array([len(lemmas)])

    def feature_names(self):
        return [self.__class__.__name__ + '_lemmas']
