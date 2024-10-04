import torchvision
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from inference import generate
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import evaluate

def train(model, 
          train_dataloader, 
          val_dataloader, 
          writer, 
          epochs, 
          validation_step, 
          log_dir,
          lr,
          eta_min,
          weight_decay):

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, eta_min=eta_min, T_max=epochs)

    train_losses = []
    val_losses = []
    model.train()

    for epoch in range(epochs):
        pbar = tqdm(train_dataloader)
        for step, batch in enumerate(pbar):
            batch = {k: v.to('cuda:0') for k, v in batch.items()}

            # Get the model prediction
            outputs = model(**batch) # Note that we pass in the labels y

            # Calculate the loss
            loss = outputs.loss
            loss.backward()
            train_losses.append(loss.item())

            # Update the model parameters with the optimizer
            optimizer.step()
            optimizer.zero_grad()
        
        lr_scheduler.step()
        writer.add_scalar('Iteration/Learning rate', optimizer.param_groups[0]['lr'], epoch)

        if (epoch + 1) % validation_step == 0:
            with torch.no_grad():
                metric = evaluate.load("accuracy")
                model.eval()
                for batch in val_dataloader:
                    batch = {k: v.to('cuda:0') for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)
                    
                    val_losses.append(outputs.loss.item())

                    logits = outputs.logits # B, L, C
                    predictions = torch.argmax(logits, dim=-1)
                    labels = batch["labels"].reshape((-1,))
                    predictions = predictions.reshape((-1,))
                    predictions = predictions[labels!=-100]
                    labels = labels[labels!=-100]
                    metric.add_batch(predictions=predictions, references=labels)

                accuracy = metric.compute()
                writer.add_scalar('Iteration/Accuracy', accuracy['accuracy'], epoch)

                epoch_train_loss = torch.tensor(train_losses).mean()
                epoch_val_loss = torch.tensor(val_losses).mean()
                writer.add_scalars('Iteration/loss', {'train':epoch_train_loss, 'val': epoch_val_loss}, epoch)

                print(f"Epoch:{epoch+1}, Train loss: {epoch_train_loss}")
                print(f"Epoch:{epoch+1}, Val loss: {epoch_val_loss}")
                max_memory_allocated = torch.cuda.max_memory_allocated(device='cuda:0')
                print("Max GPU Memory allocated:", max_memory_allocated / 10e8, "Gb")
                #overfit_data = {'Train': torch.tensor(self.train_loss).mean().item(), 'Val': torch.tensor(self.val_loss).mean().item()}
                #writer.add_scalars('Epoch/Train_vs_val_loss', overfit_data, epoch)

                torch.save(model.state_dict(), os.path.join(log_dir, 'weights.pth'))
                
    torch.save(model.state_dict(), os.path.join(log_dir, 'weights.pth'))