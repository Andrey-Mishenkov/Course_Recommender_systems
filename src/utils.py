#------------------------------------------------------------------------------------------------------------
from scipy.sparse import csr_matrix                          # Для работы с матрицами
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка

#------------------------------------------------------------------------------------------------------------
def prefilter_items(data_train, item_features):
    
    # Оставим только 5000 самых популярных товаров
    popularity = data_train.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns = {'quantity': 'n_sold'}, inplace = True)
    top_5000 = popularity.sort_values('n_sold', ascending = False).head(5000).item_id.tolist()
    
    #добавим, чтобы не потерять юзеров
    data_train.loc[~data_train['item_id'].isin(top_5000), 'item_id'] = 999999 

#     print('Топ 5000')
    
    #------------------------------------------------------------------------------------------------------------
    # Уберем самые популярные (1% по кол-ву)
    popular_top = popularity.loc[ popularity['n_sold'] > popularity['n_sold'].quantile(.99)].item_id.tolist()
    data_train = data_train.loc[~data_train['item_id'].isin(popular_top)]
    
#     print(popular_top)
    
    #------------------------------------------------------------------------------------------------------------
    # Уберем самые непопулярные (1% по кол-ву)
    popular_bottom = popularity.loc[ popularity['n_sold'] < popularity['n_sold'].quantile(.01) ].item_id.tolist()
    data_train = data_train.loc[~data_train['item_id'].isin(popular_bottom)]
    
#     print(popular_bottom)
    
    #------------------------------------------------------------------------------------------------------------
    # Уберем товары, которые не продавались за последние 12 месяцев
    df_days = data_train.groupby('item_id').agg({'day': ['max']})

    df_days.reset_index(inplace = True)
    df_days = df_days.droplevel(1, axis = 1)
    df_days.columns = ['item_id', 'day_max']

    item_not_sales = df_days.loc[df_days['day_max'] <= data_train['day'].max() - 365].item_id.tolist()
    data_train = data_train.loc[~data_train['item_id'].isin(item_not_sales)]
    
#     print(item_not_sales)
    
    #------------------------------------------------------------------------------------------------------------
    # Уберем не интересные для рекоммендаций категории (department)
    df_category = item_features[ ['department', 'item_id']].merge(data_train, on = ['item_id'], how = 'inner')

    df_category_sale = df_category.groupby('department').agg({'quantity': ['sum']})

    df_category_sale.reset_index(inplace = True)
    df_category_sale = df_category_sale.droplevel(1, axis=1)
    df_category_sale.columns = ['department', 'quantity']

    df_category_bad = df_category_sale.sort_values('quantity', ascending = False).tail(3).department.tolist()
    # df_category_bad

    items_in_bad_category = df_category.loc[df_category['department'].isin(df_category_bad)].item_id.unique()
    data_train = data_train.loc[~data_train['item_id'].isin(items_in_bad_category)]
    
#     print(items_in_bad_category)
    
    #------------------------------------------------------------------------------------------------------------
    # Уберем слишком дешевые товары (1% по цене снизу)
    # Уберем слишком дорогие товары (1% по цене сверху)
    df_prices = data_train.groupby('item_id').agg({
                                            'sales_value': ['sum'], 
                                            'quantity':    ['sum'],
                                            'item_id':     ['count']
                                            })
    df_prices.reset_index(inplace = True)
    df_prices = df_prices.droplevel(1, axis = 1)
    df_prices.columns = ['item_id', 'sales_value', 'quantity', 'count_trans']
    df_prices['price_avg'] = round(df_prices['sales_value'] / df_prices['quantity'], 6)

    prices_cheap = df_prices.loc[ df_prices['price_avg'] < df_prices['price_avg'].quantile(.01)].item_id.tolist()
    prices_rich  = df_prices.loc[ df_prices['price_avg'] > df_prices['price_avg'].quantile(.99)].item_id.tolist()
       
    #------------------------------------------------------------------------------------------------------------
    data_train = data_train.loc[~data_train['item_id'].isin(prices_cheap)]
    data_train = data_train.loc[~data_train['item_id'].isin(prices_rich)]
    #------------------------------------------------------------------------------------------------------------
    
#     print(prices_cheap)
#     print(prices_rich)
    
    return data_train

#------------------------------------------------------------------------------------------------------------
def postfilter_items():
    pass

#------------------------------------------------------------------------------------------------------------
def get_similar_items_recommendation(user, model, data, itemid_to_id, id_to_itemid, N = 5):
    """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
    
    popularity = data.loc[data['user_id'] == user].groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
    popularity.sort_values('quantity', ascending = False, inplace = True)

    popularity = popularity[popularity['item_id'] != 999999]
    popularity = popularity.groupby('user_id').head(5)

    def get_rec(model, x, N):

        recs = model.similar_items(itemid_to_id[x], N = N + 1)

        if (id_to_itemid[ recs[1][0] ] != 999999):
            top_rec = recs[1][0]
        else:
            top_rec = recs[2][0]
        
        return id_to_itemid[top_rec]

    popularity['similar_recommendation'] = popularity['item_id'].apply(lambda x: get_rec(model, x, 2))
    res = popularity['similar_recommendation'].tolist()
    
    return res

#------------------------------------------------------------------------------------------------------------
def get_similar_users_recommendation_1(user, model, user_item_matrix, userid_to_id, itemid_to_id, id_to_itemid, N = 5):
    """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
    users = model.similar_users(userid_to_id[user], N = N + 1)

    #--------------------------------------------------------------------------------------------------------------
    own = ItemItemRecommender(K = 1, num_threads = 4)                     # K - кол-во ближайших соседей
    own.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress = False)
    
    #--------------------------------------------------------------------------------------------------------------
    sparse_user_item = csr_matrix(user_item_matrix).T.tocsr()
    
    res = [id_to_itemid[rec[0]] for rec in 
            model.recommend(userid = userid_to_id[user], 
                            user_items = sparse_user_item,   # на вход user-item matrix
                            N = N, 
                            filter_already_liked_items = False, 
                            filter_items = [itemid_to_id[999999]],  # !!! 
                            recalculate_user = True)]
    return res

#------------------------------------------------------------------------------------------------------------
def get_similar_users_recommendation(user, model, user_item_matrix, userid_to_id, itemid_to_id, id_to_itemid, N = 5):
    """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
    users = model.similar_users(userid_to_id[user], N = N + 1)
    users_id = list(map(lambda x: x[0], users))
    
    #--------------------------------------------------------------------------------------------------------------
    own = ItemItemRecommender(K = 1, num_threads = 4)                     # K - кол-во ближайших соседей
    own.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress = False)
    
    #--------------------------------------------------------------------------------------------------------------
    sparse_user_item = csr_matrix(user_item_matrix).T.tocsr()
    
    def get_recommendations(user, model, sparse_user_item, N = 5):
        """Рекомендуем топ-N товаров"""
        res_1 = [id_to_itemid[rec[0]] for rec in 
                        model.recommend(userid=userid_to_id[user], 
                                        user_items=sparse_user_item,   # на вход user-item matrix
                                        N=N, 
                                        filter_already_liked_items=False, 
                                        filter_items=[itemid_to_id[999999]],  # !!! 
                                        recalculate_user=True)]
        return res_1
    
    res = [get_recommendations(user = user_id, 
                               model = own, 
                               sparse_user_item = csr_matrix(user_item_matrix).T.tocsr(), 
                               N = 1) for user_id in users_id]
    res = [item for sublist in res for item in sublist]
    
    return res
