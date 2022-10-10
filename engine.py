import torch
import torch.nn as nn
def train(data_loader, model, optimizer, device):
 # set model to training mode
 model.train()
 # go through batches of data in data loader
 for data in data_loader:
  # fetch clean and ticket_type from the dict
  clean = data["clean"]
  ticket_type = data["ticket_type"]
  # move the data to device that we want to use
  clean = clean.to(device, dtype=torch.long)
  ticket_type = ticket_type.to(device, dtype=torch.float)
  # clear the gradients
  optimizer.zero_grad()
  # make predictions from the model
  predictions = model(clean)
  # calculate the loss
  loss = nn.BCEWithLogitsLoss()(
  predictions,
  ticket_type.view(-1, 1)
  )
  # compute gradient of loss w.r.t.
  # all parameters of the model that are trainable
  loss.backward()
  # single optimization step
  optimizer.step()
def evaluate(data_loader, model, device):
 # initialize empty lists to store predictions
 # and ticket_type
    final_predictions = []
    final_targets = []
 # put the model in eval mode
    model.eval()
 # disable gradient calculation
    with torch.no_grad():
        for data in data_loader:
            clean = data["clean"]
            ticket_type = data["ticket_type"]

            clean = clean.to(device, dtype=torch.long)
            ticket_type = ticket_type.to(device, dtype=torch.float)

            predictions = model(clean)
    # move predictions and ticket_type to list
    # we need to move predictions and ticket_type to cpu too
            predictions = predictions.cpu().numpy().tolist()
            ticket_type = data["ticket_type"].cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(ticket_type)
            # return final predictions and targets
    return final_predictions, final_targets