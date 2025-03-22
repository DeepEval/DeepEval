def method():
    def is_rotation(s1, s2):
        if len(s1) != len(s2):
            return False
        return s2 in s1 + s1

    test_cases = [
        ("abcd", "abd"),
        ("hello", "ell"),
        ("whassup", "psus"),
        ("abab", "baa"),
        ("efef", "eeff"),
        ("himenss", "simen")
    ]

    results = []
    for word1, word2 in test_cases:
        results.append(is_rotation(word1, word2))

    return results

output = method()
print(output)  # Expected output: [False, True, False, True, False, True]