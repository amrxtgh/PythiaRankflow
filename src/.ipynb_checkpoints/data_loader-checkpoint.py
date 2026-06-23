import os
import pandas as pd

class DataLoader:
	"""
	Loads MovieLens 1M dataset in the clean format for recommendation research.
	"""
	def __init__(self, data_path):
		self.data_path = data_path
	
	def load_rating(self):
		return pd.read_csv(
			os.path.join(self.data_path, "ratings.dat"),
			sep="::",
			engine="python",
			names=["user_id", "movie_id", "rating", "timestamp"],
			encoding="latin-1"
		)
	def load_movies(self):
		return pd.read_csv(
			os.path.join(self.data_path, "movies.dat"),
			sep="::",
			engine="python",
			names=["movie_id", "title", "genre"],
			encoding="latin-1"
		)
	def load_users(self):
		return pd.read_csv(
			os.path.join(self.data_path, "users.dat"),
			sep="::",
			engine="python",
			names=["user_id", "gender", "age", "occupation", "zip_code"],
			encoding="latin-1"
		)

	def load_full_dataset(self):
		ratings = self.load_rating()
		movies = self.load_movies()
		users = self.load_users()

		data = ratings.merge(movies, on="movie_id")
		data = data.merge(users, on="user_id")

		return data


	
