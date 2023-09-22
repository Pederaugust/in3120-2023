def f(s: str) -> None:
    suffix_array = sorted([(s[i:], i) for i in range(len(s))], key=lambda x: x[0])
    return suffix_array

# print(f(input()))