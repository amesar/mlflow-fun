
# From https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html

import torch
import mlflow
import mlflow.pytorch 

# X data
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# Y data with its expected value: labels
y_data = torch.Tensor([[2.0], [4.0], [6.0]])
# Partial Model example modified from Sung Kim
# https://github.com/hunkim/PyTorchZeroToAll

class Model(torch.nn.Module):
   def __init__(self):
      super(Model, self).__init__()
      self.linear = torch.nn.Linear(1, 1)  # One in and one out
   def forward(self, x):
       y_pred = self.linear(x)
       return y_pred
# our model
model = Model()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# Training loop
epochs = 500
epochs = 5
for epoch in range(epochs):
   # Forward pass: Compute predicted y by passing x to the model
   y_pred = model(x_data)
   # Compute and print loss
   loss = criterion(y_pred, y_data)
   print(epoch, loss.data[0])
   #Zero gradients, perform a backward pass, and update the weights.
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()

# After training
for hv in [4.0, 5.0, 6.0]:
    hour_var = torch.Tensor([[hv]])
    y_pred = model(hour_var)
    print("predict (after training)",  hv, model(hour_var).data[0][0])

# log the model
experiment_id = 0
with mlflow.start_run(experiment_id=experiment_id) as run:
    mlflow.log_param("epochs", epochs)
    mlflow.pytorch.log_model(model, "model")
