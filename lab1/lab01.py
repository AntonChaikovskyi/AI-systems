class Neuron:
    def __init__(self, w1, w2, bias):
        self.w1 = w1
        self.w2 = w2
        self.bias = bias

    def feed_forward(self, x1, x2):
        total = (x1 * self.w1) + (x2 * self.w2) + self.bias
        return 1 if total > 0 else 0


neuron_OR = Neuron(w1=1, w2=1, bias=-0.5)
neuron_AND = Neuron(w1=1, w2=1, bias=-1.5)
neuron_XOR = Neuron(w1=1, w2=-1, bias=-0.5)

inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

print("-" * 45)
print(f"{'Вхід x1':<10} | {'Вхід x2':<10} | {'OR':<5} | {'AND':<5} | {'XOR (Фінал)':<10}")
print("-" * 45)

for x1, x2 in inputs:
    res_or = neuron_OR.feed_forward(x1, x2)
    res_and = neuron_AND.feed_forward(x1, x2)
    res_final = neuron_XOR.feed_forward(res_or, res_and)

    print(f"{x1:<10} | {x2:<10} | {res_or:<5} | {res_and:<5} | {res_final:<10}")

print("-" * 45)