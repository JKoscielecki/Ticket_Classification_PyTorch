import torch
class IMDBDataset:
  def __init__(self, clean, ticket_type):
    # """
    # :param clean: this is a numpy array
    # :param ticket_type: a vector, numpy array
    # """
    self.clean = clean
    self.ticket_type = ticket_type
  def __len__(self):
    # returns length of the dataset
    return len(self.clean)

  def __getitem__(self, item):
    # for any given item, which is an int,
    # return clean and ticket_type as torch tensor
    # item is the index of the item in concern
    clean = self.clean[item, :]
    ticket_type = self.ticket_type[item]
    return {

    "clean": torch.tensor(clean, dtype=torch.long),
    "ticket_type": torch.tensor(ticket_type, dtype=torch.float)
    }