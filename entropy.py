import torch

#Entropy
def entropy(Y):
    """
    Shanon Entropy
    """
    unique, count = torch.unique(Y, return_counts=True, dim=0)
    prob = count/len(Y)
    en = torch.sum((-1)*prob*torch.log2(prob))
    return en

#Joint Entropy
def joint_entropy(Y,X):
    """
    H(Y;X)
    """
    YX = torch.hstack([Y,X])
    return entropy(YX)

#Conditional Entropy
def conditional_entropy(Y, X):
    """
    H(Y|X) = H(Y;X) - H(X)
    """
    return joint_entropy(Y, X) - entropy(X)

#Mutual Information
def mutual_info(Y, X):
    """
    I(Y;X) = H(Y) - H(Y|X)
    """
    return entropy(Y) - conditional_entropy(Y,X)

if __name__ == "__main__":
    print(entropy(torch.tensor([[1, 2, 3], [1,2,3], [2,4,6], [2,4,6]])))