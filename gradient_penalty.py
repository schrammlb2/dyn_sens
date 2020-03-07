import torch
import pdb

def gradient_penalty(critic, state, action=None, epsilon = 1e-4):
    state_copy = state.clone().detach().requires_grad_(True)
    	#Dummy variable to collect the gradient
    # state_dummy = torch.zeros_like(state, requires_grad=True)
    if action is not None:
        new_values = critic(state_copy, action.detach())
    else: 
        new_values = critic(state_copy)

    new_values.mean().backward(retain_graph=True)
    grads = state_copy.grad.view(state.shape[0], -1)
    grad_norms = torch.mean(grads**2, dim=1)**.5
    grad_norms = grad_norms.view_as(new_values)
    grad_penalty = epsilon*grad_norms
    return new_values - grad_penalty


def gradient_penalty(critic, state, action=None, epsilon = 1e-4):
    state_copy = state.clone().detach().requires_grad_(True)
        #Dummy variable to collect the gradient
    # state_dummy = torch.zeros_like(state, requires_grad=True)
    if action is not None:
        action_copy = action.clone().detach().requires_grad_(True)
        new_values = critic(state_copy, action_copy)
    else: 
        new_values = critic(state_copy)

    new_values.mean().backward(retain_graph=True)
    grads = state_copy.grad.view(state.shape[0], -1)
    grad_norms = torch.mean(grads**2, dim=1)**.5
    grad_norms = grad_norms.view_as(new_values)
    grad_penalty = epsilon*grad_norms
    if action is not None:
        grads = action_copy.grad.view(action.shape[0], -1)
        grad_norms = torch.mean(grads**2, dim=1)**.5
        grad_norms = grad_norms.view_as(new_values)
        grad_penalty += epsilon*grad_norms

    return new_values - grad_penalty


def action_gradient(critic, state, action, epsilon = 1e-4):
    state_copy = state.clone().detach()
        #Dummy variable to collect the gradient
    action_copy = action.clone().detach().requires_grad_(True)
    new_values = critic(state_copy, action_copy)
    
    new_values.mean().backward(retain_graph=True)
    
    grads = action_copy.grad.view(action.shape[0], -1)
    grad_norms = torch.mean(grads**2, dim=1)**.5
    grad_norms = grad_norms.view_as(new_values)
    normal_grad = action_copy.grad/grad_norms

    return normal_grad*epsilon


def gradient_penalty_with_policy(critic, state, policy, epsilon = 1e-4):
    state_copy = state.clone().detach().requires_grad_(True)
        #Dummy variable to collect the gradient
    # state_dummy = torch.zeros_like(state, requires_grad=True)
    if action is not None:
        new_values = critic(state_copy, policy(state_copy))
    else: 
        new_values = critic(state_copy)

    # new_values.mean().backward(retain_graph=True)
    new_values.mean().backward()
    grads = state_copy.grad.view(state.shape[0], -1)
    grad_norms = torch.mean(grads**2, dim=1)**.5
    grad_norms = grad_norms.view_as(new_values)
    grad_penalty = epsilon*grad_norms
    return new_values - grad_penalty