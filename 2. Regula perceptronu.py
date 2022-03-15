'''η := 0.1		# współczynnik nauczania
w := [ -1, -1, -1 ]	# wektor wag (do x0, x1, x2) x0 – bias
i := 1   		# licznik iteracji
error := -1		# błąd klasyfikacji
WHILE (error != 0 AND i <= 1000)
	SHUFFLE (p, t)          # losowanie kolejności wektorów p oraz t (musimy zachować kolejność między nimi)
	FOR (j := 1 … n)	# zgodnie z tabelką n = 8
		net := pj ∙ w	# iloczyn skalarny (suma przemnożonych wejść przez wagi)
		     = pj,0 w0 + pj,1 w1 + pj,2 w2
		y := f(net)                     # wywołanie funkcji progowej bipolarnej
		r := tj – y
		∆w := η r pj		# # η – liczba, r – liczba, pj – wektor, czyli ∆w wektorem zmiany wag
		w = w + Δw
		----- RYSOWANIE WYKRESU -----
	END
	error = 0                       # zerowanie błędu
	FOR (j := 1 … n)		# zgodnie z tabelką n = 8
		net := pj ∙ w	# iloczyn skalarny (suma przemnożonych wejść przez wagi)
		     = pj,0 w0 + pj,1 w1 + pj,2 w2
		y := f(net)            # wywołanie funkcji progowej bipolarnej
		IF (y != tj)
			error = error + 1
		END
	END
	i = i + 1
END'''
import random
import numpy as np
import matplotlib.pyplot as plt
def neuron(inputs, weights):
    net = sum([a*b for a, b in zip(inputs, weights)])
    if net >= 0:
        y = 1
    else:
        y = -1
    return y
def plot_dots(data):
    for i in data:
        if i[1] == -1 :
            c = 'blue'
        else:
            c = 'yellow'
        plt.scatter(i[0][1], i[0][2], c=c)

def plot_line(weights, eta, last):
# 0 = x1*w1 + x2*w2 + wb
# 0 = p[0]*weights[1] + p[1]*weights[2] + weights[0]
# p[1]*weights[2] = -(p[0]*weights[1]) -  weights[0]
#  p[1] = (-(p[0]*weights[1]) -  weights[0])/weights[2]
    x = np.linspace(-5,5)
    y = (-(x * weights[1]) - weights[0])/weights[2]
    col = (np.random.random(), np.random.random(), np.random.random())
    line = plt.plot(x, y, c =  col)
    plt.xlabel('x1', color='#1C2833')
    plt.ylabel('x2', color='#1C2833')
    plt.pause(eta*5)
    if last == False:
        l = line.pop(0)
        l.remove()

def linear_train(data, eta):
    weights = []
    for _ in range (len (data[0][0])):
        weights.append(-1)
    i = 0
    error = True
    while error == True and i < 1000:
        random.shuffle(data)
        r_holder = [0]
        k = 0
        for j in data:
            p = j[0]
            t = j[1]
            y = neuron(p, weights)
            r = t - y
            if r != 0:
                r_holder.append(1)
            r_holder.append(r)
            delta = eta * r * np.array(p)
            weights += delta
            print("iteracja ", i, ' (probka' ,k ,' z ', len(data)-1, ").")
            #rysowanie wykresu
            plot_line(weights, eta, False)
            plt.draw()
            k += 1
        if sum(r_holder)== 0:
            error = False
            plot_line(weights, eta, True)
        i += 1



def generate_vectors(y, *x):
    bias = []
    for _ in range(len(y)):
        bias.append(1)
    x = list(x)
    x.insert(0, bias)
    p = list(zip(*x))
    t = y
    data = list(zip(p, t))
    return data

#training data
x1_inputs = [-3, -2, 0, 2, -2, 0, 2, 3]
x2_inputs = [4, 1, 1, 2, -4, -2, 1, -4]
y_outputs = [-1, -1, -1, -1, 1, 1, 1, 1]
eta = float(input('podaj wartośc eta: '))
data = generate_vectors(y_outputs, x1_inputs, x2_inputs)
plot_dots(data)
linear_train(data, eta)
plt.show()

