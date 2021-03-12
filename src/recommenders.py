#------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, weighting = True):
        
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        
        self.data = data
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data):
        
        user_item_matrix = pd.pivot_table(data, 
                                          index='user_id', columns='item_id', 
                                          values='quantity', # Можно пробовать другие варианты
                                          aggfunc='count', 
                                          fill_value=0
                                         )

        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit

#         print(user_item_matrix.head(3))
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors        = n_factors, 
                                        regularization = regularization,
                                        iterations     = iterations,  
                                        num_threads    = num_threads)
#         model.fit(csr_matrix(self.user_item_matrix).T.tocsr())
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return model

#     @staticmethod
    def get_similar_items_recommendation(self, user, N = 5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
    
        popularity = self.data.loc[self.data['user_id'] == user].groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        popularity.sort_values('quantity', ascending = False, inplace = True)

        popularity = popularity[popularity['item_id'] != 999999]
        popularity = popularity.groupby('user_id').head(5)

        def get_rec(model, x, N):

            recs = model.similar_items( self.itemid_to_id[x], N = N + 1)

            if ( self.id_to_itemid[ recs[1][0] ] != 999999):
                top_rec = recs[1][0]
            else:
                top_rec = recs[2][0]

            return self.id_to_itemid[top_rec]

        popularity['similar_recommendation'] = popularity['item_id'].apply(lambda x: get_rec(self.model, x, 2))
        res = popularity['similar_recommendation'].tolist()

        return res

#         assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
#         return res
    
#     @staticmethod
    def get_similar_users_recommendation(self, user, N = 5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        users = self.model.similar_users( self.userid_to_id[user], N = N + 1)
        users_id = list(map(lambda x: x[0], users))

        #--------------------------------------------------------------------------------------------------------------
#         own = ItemItemRecommender(K = 1, num_threads = 4)                     # K - кол-во ближайших соседей
#         own.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress = False)

        #--------------------------------------------------------------------------------------------------------------
        sparse_user_item = csr_matrix(self.user_item_matrix).T.tocsr()

        def get_recommendations(user, model, sparse_user_item, N = 5):
            """Рекомендуем топ-N товаров"""
            res_1 = [self.id_to_itemid[rec[0]] for rec in 
                            self.model.recommend(userid = self.userid_to_id[user], 
                                                user_items = sparse_user_item,   # на вход user-item matrix
                                                N = N, 
                                                filter_already_liked_items = False, 
                                                filter_items = [self.itemid_to_id[999999]],  # !!! 
                                                recalculate_user = True)]
            return res_1

        res = [get_recommendations(user = user_id, 
                                   model = self.own_recommender, 
                                   sparse_user_item = csr_matrix(self.user_item_matrix).T.tocsr(), 
                                   N = 1)
               for user_id in users_id]
        res = [item for sublist in res for item in sublist]

        return res

#         assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
#         return res