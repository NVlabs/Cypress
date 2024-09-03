import torch
import torch.nn.functional as F
import torch.nn.functional

# Define a differentiable cost function for a sequence of choices
def cost_function(choices):
    ideal_choices = torch.tensor([2, 1, 3, 0, 3], dtype=torch.float32)
    return torch.sum((choices - ideal_choices) ** 2)

# Initialize logits randomly for 5 choices, each with 4 options
logits = torch.randn(5, 4, requires_grad=True)

# Define Gumbel-Softmax function
def gumbel_softmax(logits, tau=1.0):
    gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
    y = logits + gumbels
    return F.softmax(y / tau, dim=-1)

# Set temperature parameter
tau = 1.0

# Define optimizer
optimizer = torch.optim.SGD([logits], lr=0.1)

# Number of gradient descent steps
num_steps = 1000

for step in range(num_steps):
    # Zero the gradients
    optimizer.zero_grad()
    
    # Sample using Gumbel-Softmax for each choice
    y = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True)
    
    # Convert softmax probabilities to discrete choices
    index_tensor = torch.arange(4).unsqueeze(0).expand_as(y)
    choices = torch.sum(y * index_tensor, dim=1)

    # Multiply by pi/2 to get the angle in radians
    theta = choices * (math.pi / 2)
    
    # Compute the cost for the sequence of choices
    cost = cost_function(choices)
    
    # Compute the gradients and perform a gradient descent step
    cost.backward()
    optimizer.step()
    
    # Print the current state
    print(f"Step {step+1}, Choices: {choices.data.numpy()}, Cost: {cost.item()}, Logits: {logits.data.numpy()}")

# Final chosen options
final_choices = torch.argmax(gumbel_softmax(logits, tau), dim=1).data.numpy()
# final_choices = torch.sum(gumbel_softmax(logits, tau) * index_tensor, dim=1).data.numpy().round()
print(f"Final chosen options: {final_choices}")
