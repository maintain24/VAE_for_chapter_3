# -*- coding:utf-8 -*-

"""
采用动态规划，用于计算两个smiles序列的准确率，
"""
def align_smiles(smiles1: str, smiles2: str) -> int:
    """
    Align two SMILES strings and return the number of matching characters.
    This uses a dynamic programming approach to find the optimal alignment.

    :param smiles1: First SMILES string
    :param smiles2: Second SMILES string
    :return: Number of matching characters in the optimal alignment
    """
    len1, len2 = len(smiles1), len(smiles2)
    # Creating a matrix for dynamic programming
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Fill the matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if smiles1[i - 1] == smiles2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # The maximum number of matching characters
    return dp[len1][len2]

def calculate_accuracy(smiles1: str, smiles2: str) -> float:
    """
    Calculate the accuracy of predicted SMILES string compared to the target SMILES string.

    :param smiles1: Target SMILES string
    :param smiles2: Predicted SMILES string
    :return: Accuracy as a percentage
    """
    matches = align_smiles(smiles1, smiles2)
    max_length = max(len(smiles1), len(smiles2))
    return (matches / max_length) * 100 if max_length > 0 else 0

# Example SMILES strings
smiles_target = "O=CCO[C@H](CO)[C@@H](O)C=O"
smiles_predicted = "O=CO([C@H]O[C@H][C@H](O"

# Calculate accuracy
accuracy = calculate_accuracy(smiles_target, smiles_predicted)
print('准确率：', accuracy)