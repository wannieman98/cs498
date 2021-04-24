import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from torch.nn.functional import mse_loss

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    
    loss = None
    zeros = torch.zeros_like(logits_fake, requires_grad=True).detach()
    ones = torch.ones_like(logits_real, requires_grad=True)
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    log_real = bce_loss(logits_real, ones, reduction='mean')
    log_fake = bce_loss(logits_fake, zeros, reduction='mean')

    loss = log_real + log_fake
    ##########       END      ##########
    
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    logits_target = torch.ones_like(logits_fake, requires_grad=True)

    return bce_loss(input=logits_fake, target=logits_target, reduction='mean')
    ##########       END      ##########
    
    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    ones = torch.ones_like(scores_real, requires_grad=True)
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    e_real = torch.div(torch.mean(torch.square(torch.sub(scores_real, ones))), 2)
    e_fake = torch.div(torch.mean(torch.square(scores_fake)), 2)
    loss = e_real + e_fake
    ##########       END      ##########
    
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    ones = torch.ones_like(scores_fake, requires_grad=True)
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    loss = torch.div(torch.mean(torch.square(torch.sub(scores_fake, ones))), 2)
    ##########       END      ##########
    
    return loss
