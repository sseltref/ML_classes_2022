import matplotlib.pyplot as plt
def neuron(inputs, weights, T):
    net = sum([a*b for a, b in zip(inputs, weights)])
    if net >= T:
        y = 1
    else:
        y = 0
    return y
def or_gate(inputs):
    weights = [1, 1]
    y = neuron(inputs, weights, 1)
    return y
def not_gate(inputs):
    weights = [-1]
    y = neuron(inputs, weights, 0)
    return y
###
def nor_gate(inputs):
    y = not_gate([or_gate(inputs)])
    return y
###
def nand_gate(inputs):
    y = or_gate([not_gate([inputs[0]]), not_gate([inputs[1]])])
    return y
def plot(x1, x2, y):
    # 1-czerwony, 0 - czarny
    if y == 1:
        c = "red"
    else:
        c = "black"
    plt.scatter(x1, x2, c=c)


def generator(inputs):
    i = 0
    print("nor gate")
    print("##########")
    plt.figure(1)
    while i < len(inputs):
        y = nor_gate(inputs[i])
        print(inputs[i], " -----> ", y)
        plot(inputs[i][0],inputs[i][1], y)
        i+=1
    plt.title("bramka nor")
    i = 0
    plt.figure(2)
    print("nand gate")
    print("##########")
    while i < len(inputs):
        y = nand_gate(inputs[i])
        print(inputs[i], " -----> ", y)
        plot(inputs[i][0], inputs[i][1], y)
        i+=1
    plt.title("bramka nand")



inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
generator(inputs)
plt.show()
