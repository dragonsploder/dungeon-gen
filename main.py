# Dungeon Gen MLP
import torch
import torch.nn.functional as F

# Read input

fdata = 'data.txt'
f = open(fdata, 'r').read().splitlines()
data_len = len(f)
print(data_len)

sample_len = 12
encoding_dim = 2
hidden_layer_size = 100

batch_size = 30

trainings = 10000

learning_rate = 0.01

size = 3

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

input_mat = torch.tensor(input_samples)
output_mat = torch.tensor(output_samples)

# Build model
embedding_mat = torch.randn((unique_inputs, encoding_dim))

weights_one = torch.randn((sample_len * encoding_dim, hidden_layer_size))
bias_one = torch.randn((hidden_layer_size))

weights_two = torch.randn((hidden_layer_size, unique_inputs))
bias_two = torch.randn((unique_inputs))

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

    # Backward
    for p in parameters:
        p.grad = None

    loss.backward()

    # Update
    for p in parameters:
        p.data += -learning_rate * p.grad

print(loss.item())

output_string = [[' ' for i in range(22)] for j in range(80)]

for y in range(22):
    for x in range(80):
        # print(y, x)
        input_str = "{:02}".format(y) + "{:02}".format(x)
        for i in range(size - 1,-1,-1):
            for j in range(size - 1,-1,-1):
                if i == 0 and j == 0:
                    break
                elif x - i < 0 or y - j < 0:
                    input_str += ' '
                else:
                    input_str += output_string[x - i][y - j]

        # print(input_str)

        embedded = embedding_mat[torch.tensor([char_to_int[i] for i in input_str])]
        hidden_layer = torch.tanh(embedded.view(1, -1) @ weights_one + bias_one)
        logits = hidden_layer @ weights_two + bias_two
        probs = F.softmax(logits, dim=1)
        char_index = torch.multinomial(probs, num_samples=1).item()
        output_char = int_to_char[char_index]
        output_string[x][y] = output_char
        print(output_char, end='')
    print()