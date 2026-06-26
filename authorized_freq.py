divisors = []

for a in range(6):
    for b in range(6):
        x = (2**a) * (5**b)
        if x <= 1500:
            divisors.append(x)

divisors = sorted(set(divisors))

print(divisors)
print("Nombre de solutions :", len(divisors))