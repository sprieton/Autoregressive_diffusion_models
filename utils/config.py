# CONFIG FILE


# ------- ENVIROMENT -------
results_dir = "./results"

# ------- CONSTANTS -------
img_w =             28           # image MINST 28x28=784
img_h =             28
img_dim =           img_w*img_h

# ------- TRAIN PARAMETERS -------
num_epochs =        30
bach_size =         128
num_clases =        2

# ------- MODEL HYPERPARAMETERS -------
hidden_dim =        512
time_emb_dim =      64         # dimension of time embedding
num_heads =         4   
num_trans_layers=   2