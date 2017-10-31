import numpy as np

class KMeans(object):

	def __init__(self, data, k=5, n_iter=10, random_state=47):

		self.k = k
		self.data = data
		self.cluster_centers = np.zeros((k, data.shape[1]))
		self.sum_cluster_centers = np.zeros((k, data.shape[1]))
		self.n_iter = n_iter
		self.assigned_clusters = np.zeros(data.shape[0])
		self.shuffled_index = np.arange(data.shape[0])
		self.completed_iter = 0
		np.random.seed(random_state)

	def fit(self):

		for i in range(self.n_iter):
			self._run_iteration()
			self.cluster_centers = self.sum_cluster_centers / self.n_iter

	def predict(self, data):

		predictions = self._minimum_distance(data)
		for i in range(self.k):
			self.cluster_centers[i] = np.mean(data[np.where(predictions==i)], axis=0)
		return predictions

	def _run_iteration(self):

		self._update_cluster_centers(initialize=True)
		self._assign_clusters()

	def _update_cluster_centers(self, initialize=False, predict=False):

		if initialize:      # Initially, the 'k' clusters are initialized by 'k' random points from the data
			np.random.shuffle(self.shuffled_index)
			self.cluster_centers = self.data[self.shuffled_index[:self.k]]

		else:
			for i in range(self.k):
				self.cluster_centers[i] = np.mean(self.data[np.where(self.assigned_clusters==i)], axis=0)
			if not predict:
				self._assign_clusters()

	def _assign_clusters(self):

		argmin_distance = self._minimum_distance(self.data)
		if (self.assigned_clusters == argmin_distance).all():
			self._return_results()
		else:
			self.assigned_clusters = argmin_distance
			self._update_cluster_centers()

	def _minimum_distance(self, data):

		num = data.shape[0]                # No. of data points
		distance_matrix = np.zeros(num * self.k).reshape(num, self.k)
		for i, cluster in enumerate(self.cluster_centers):
			distance = np.apply_along_axis(self._euclidean_distance, 1, data, cluster)
			distance_matrix[:,i] = distance
		distance_matrix = np.array(distance_matrix)
		return distance_matrix.argmin(axis=1)

	def _euclidean_distance(self, x1, x2):

		# Returns the Euclidean's Distance between two points, x1 and x2
		return np.sqrt(np.sum(np.square(x1 - x2)))

	def _return_results(self):

		self.completed_iter += 1
		self.sum_cluster_centers += self.cluster_centers
		self._reset()

	def _reset(self):

		self.cluster_centers = np.zeros((self.k, self.data.shape[1]))
		self.assigned_clusters = np.zeros(self.data.shape[0])
		self.shuffled_index = np.arange(self.data.shape[0])