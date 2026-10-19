from collections.abc import Iterable

def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	def list_wrapper(values):
		if isinstance(values, Iterable):
			return list(values)
		return [values]

	yt_list = list_wrapper(y_true)
	yp_list = list_wrapper(y_pred)

	if len(yt_list) != len(yp_list):
		raise ValueError("Inputs are not same length.")

	losses = []
	for yt, yp in zip(yt_list, yp_list):
		e = yp - yt
		if abs(e) <= delta:
			huber = 0.5 * e**2 
		else:
			huber = delta * (abs(e) - 0.5 * delta)
			
		losses.append(huber)
	
	total_loss = sum(losses)
	huber_avg = total_loss/len(yt_list)

	return huber_avg