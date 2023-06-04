import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def colfil(user_id, ratings, n):
	ratings = pd.DataFrame(ratings, columns=['user_id', 'product_id','rating'])

	user_item_matrix = ratings.pivot_table(index='user_id', columns='product_id', values='rating')
	user_similarity_matrix = pd.DataFrame(cosine_similarity(user_item_matrix.fillna(0)))
	user_similarity_matrix.index = user_item_matrix.index
	user_similarity_matrix.columns = user_item_matrix.index

	similarity_scores = user_similarity_matrix.loc[user_id]
	similar_users = similarity_scores.sort_values(ascending=False).index.tolist()
	recommendations = []
	for item in user_item_matrix.columns:
		if pd.isnull(user_item_matrix.loc[user_id, item]):
			ratings = user_item_matrix.loc[similar_users, item]
			ratings = ratings[ratings.notnull()]
			if len(ratings) > 0:
				recommendation_score = ratings.mean()
				recommendations.append((item, recommendation_score))
	recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:n]
	product_id = [product[0] for product in recommendations]
	return product_id

