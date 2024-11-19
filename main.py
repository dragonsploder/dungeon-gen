# Dungeon Gen MLP
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 

#gpu
device = "cuda"

# Read input

fdata = 'data.txt'
f = open(fdata, 'r').read().splitlines()
data_len = len(f)
print(data_len)

sample_len = 14
encoding_dim = 25
hidden_layer_size = 40

batch_size = 100

trainings = 500000

learning_rate = 0.001

chars = sorted(list(set(''.join(f))))
unique_inputs = len(chars)
char_to_int = {s:i for i,s in enumerate(chars)}
int_to_char = {i:s for s,i in char_to_int.items()}

# input [12 char] -> encoding [2 dim] -> hidden layer (100 nodes) -> softmax (output) 
input_samples = []
output_samples = []
for line in f:
    sample = line.split('}')
    input_sample = list(sample[0])
    input_samples.append([char_to_int[ch] for ch in input_sample])
    output_samples.append(char_to_int[sample[1]])

input_mat = torch.tensor(input_samples).to(device)
output_mat = torch.tensor(output_samples).to(device)

# Build model
embedding_mat = torch.randn((unique_inputs, encoding_dim)).to(device)

weights_one = (torch.randn((sample_len * encoding_dim, hidden_layer_size)) * 0.02).to(device)
bias_one = (torch.randn((hidden_layer_size)) * 0.01).to(device)

weights_two = (torch.randn((hidden_layer_size, unique_inputs)) * 0.01).to(device)
bias_two = (torch.randn((unique_inputs)) * 0).to(device)

parameters = [embedding_mat, weights_one, bias_one, weights_two, bias_two]
for p in parameters:
  p.requires_grad = True


# Traning Loop
loss = 0
for i in range(trainings):
    batch = torch.randint(0, input_mat.shape[0], (batch_size,))

    # Forward
    embedded = embedding_mat[input_mat[batch]]
    hidden_layer = torch.tanh(embedded.view(-1, sample_len * encoding_dim) @ weights_one + bias_one)
    logits = hidden_layer @ weights_two + bias_two
    loss = F.cross_entropy(logits, output_mat[batch])
    #print(loss.item())
    if i % 1000 == 0:
        print(i, loss.item())

    if i == int(trainings * 0.80):
        learning_rate /= 10

    # Backward
    for p in parameters:
        p.grad = None

    loss.backward()

    # Update
    for p in parameters:
        p.data += -learning_rate * p.grad

print("Final:", loss.item())
with open("loss_stats.txt", "a") as f:
    f.write("Loss: " + str(loss.item()) + " Trainings:" + str(trainings) + " Batch:" + str(batch_size) + " Hidden:" + str(hidden_layer_size) + " Embed:" + str(encoding_dim) + "\n")


output_string = [[' ' for i in range(80)] for j in range(22)]

for _ in range(10):
    print("Dungeon:")
    rows=22
    cols=80 
    for i in range(rows):
        for j in range(cols):
            input_str = "{:02}".format(i) + "{:02}".format(j)
            for y in range(-2,1):
                for x in range(-2,2):
                    if y == 0 and x == 0:
                        break
                    elif i + y < 0 or j + x < 0 or i + y >= rows or j + x >= cols:
                        input_str += ' '
                    else:
                        input_str += output_string[i + y][j + x]

            # print(input_str)

            embedded = embedding_mat[torch.tensor([char_to_int[i] for i in input_str])]
            hidden_layer = torch.tanh(embedded.view(1, -1) @ weights_one + bias_one)
            logits = hidden_layer @ weights_two + bias_two
            probs = F.softmax(logits, dim=1)
            char_index = torch.multinomial(probs, num_samples=1).item()
            output_char = int_to_char[char_index]
            if output_char.isdigit():
                output_char = ' '
            output_string[i][j] = output_char
            print(output_char, end='')
        print()

# https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb
# plt.figure(figsize=(8,8))
# plt.scatter(embedding_mat[:,0].data, embedding_mat[:,1].data, s=200)
# for i in range(embedding_mat.shape[0]):
#     plt.text(embedding_mat[i,0].item(), embedding_mat[i,1].item(), int_to_char[i], ha="center", va="center", color='white')
# plt.grid('minor')