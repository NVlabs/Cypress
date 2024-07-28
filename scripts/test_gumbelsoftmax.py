import torch
import torch.nn.functional as F

# Define a differentiable cost function
def cost_function(choice):
    # since this does not need to be differentiable to logits
    # we can use external functions, e.g. call HLS???
    return (choice - 2) ** 2

# Initialize logits randomly
# logits = torch.randn(4, requires_grad=True)
logits = torch.tensor([0.0, 0.0, 1.0, 0.0], requires_grad=True)

# Define Gumbel-Softmax function
def gumbel_softmax(logits, tau=1.0):
    gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
    y = logits + gumbels
    return F.softmax(y / tau, dim=-1)

# Set temperature parameter
tau = 0.5

# Define optimizer
optimizer = torch.optim.SGD([logits], lr=0.1)

# Number of gradient descent steps
num_steps = 1000

for step in range(num_steps):
    # Zero the gradients
    optimizer.zero_grad()
    
    # Sample using Gumbel-Softmax
    y = gumbel_softmax(logits, tau)
    
    # Compute the cost for each option
    # interesting example: this means the gradients does not need to propagate through the cost tensor
    cost_tensor = torch.tensor([cost_function(i) for i in range(4)], dtype=torch.float32)
    
    # Compute the expected cost by weighting with the softmax output
    expected_cost = torch.sum(y * cost_tensor)
    
    # Compute the gradients and perform a gradient descent step
    expected_cost.backward()
    optimizer.step()
    
    # Print the current state
    print(f"Step {step+1}, Logits: {logits.data.numpy()}, Expected Cost: {expected_cost.item()}")

# Final chosen option
final_choice = torch.argmax(gumbel_softmax(logits, tau)).item()
print(f"Final chosen option: {final_choice}")
