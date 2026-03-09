def compute_confident_joint(label_pairs):
    confident_joint = {}
    for true_label, observed_label in label_pairs:
        key = (true_label, observed_label)
        if key in confident_joint:
            confident_joint[key] += 1
        else:
            confident_joint[key] = 1
    return confident_joint

if __name__ == "__main__":
    # Example usage
    label_pairs = [('cat', 'cat'), ('cat', 'dog'), ('dog', 'dog'), ('dog', 'cat'), ('dog', 'dog')]
    result = compute_confident_joint(label_pairs)
    print(result)