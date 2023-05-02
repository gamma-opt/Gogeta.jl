

# some raw data definitions for testing etc.
x_train, y_train = MNIST(split=:train)[:]
x_test, y_test = MNIST(split=:test)[:]
y_train_oh = onehotbatch(y_train, 0:9)
x_train_flatten = flatten(x_train)
x_test_flatten = flatten(x_test)

adversarial_L1 = readfile("adversarial_L1_1-12000.jld")
adversarial_L2 = readfile("adversarial_L2_1-12000.jld")

# mnist model for original train set
jm_raw, accuracy_raw = train_mnist_nn(x_train, y_train, x_test, y_test)
println("Accuracy, raw train set: ", accuracy_raw, "\n")


# ### neural nets for added L1-norm ###
# # mnist model for added 1% L1-norm
# x_add1_L1, y_add1_L1 = add_adversarials(adversarial_L1, 600)
# jm_add1_L1, acc_add1_L1 = train_mnist_nn(x_add1_L1, y_add1_L1, x_test, y_test)
# println("Accuracy, added 1% of L1-norm: ", acc_add1_L1, "\n")

# # mnist model for added 5% L1-norm
# x_add5_L1, y_add5_L1 = add_adversarials(adversarial_L1, 3000)
# jm_add5_L1, acc_add5_L1 = train_mnist_nn(x_add5_L1, y_add5_L1, x_test, y_test)
# println("Accuracy, added 5% of L1-norm: ", acc_add5_L1, "\n")

# # mnist model for added 10% L1-norm
# x_add10_L1, y_add10_L1 = add_adversarials(adversarial_L1, 6000)
# jm_add10_L1, acc_add10_L1 = train_mnist_nn(x_add10_L1, y_add10_L1, x_test, y_test)
# println("Accuracy, added 10% of L1-norm: ", acc_add10_L1, "\n")

# mnist model for added 20% L1-norm
x_add20_L1, y_add20_L1 = add_adversarials(adversarial_L1, 12000)
jm_add20_L1, acc_add20_L1 = train_mnist_nn(x_add20_L1, y_add20_L1, x_test, y_test)
println("Accuracy, added 20% of L1-norm: ", acc_add20_L1, "\n")


### neural nets for added L2-norm ###
# # mnist model for added 1% L2-norm
# x_add1_L2, y_add1_L2 = add_adversarials(adversarial_L2, 600)
# jm_add1_L2, acc_add1_L2 = train_mnist_nn(x_add1_L2, y_add1_L2, x_test, y_test)
# println("Accuracy, added 1% of L2-norm: ", acc_add1_L2, "\n")

# # mnist model for added 5% L2-norm
# x_add5_L2, y_add5_L2 = add_adversarials(adversarial_L2, 3000)
# jm_add5_L2, acc_add5_L2 = train_mnist_nn(x_add5_L2, y_add5_L2, x_test, y_test)
# println("Accuracy, added 5% of L2-norm: ", acc_add5_L2, "\n")

# # mnist model for added 10% L2-norm
# x_add10_L2, y_add10_L2 = add_adversarials(adversarial_L2, 6000)
# jm_add10_L2, acc_add10_L2 = train_mnist_nn(x_add10_L2, y_add10_L2, x_test, y_test)
# println("Accuracy, added 10% of L2-norm: ", acc_add10_L2, "\n")

# mnist model for added 20% L2-norm
x_add20_L2, y_add20_L2 = add_adversarials(adversarial_L2, 12000)
jm_add20_L2, acc_add20_L2 = train_mnist_nn(x_add20_L2, y_add20_L2, x_test, y_test)
println("Accuracy, added 20% of L2-norm: ", acc_add20_L2, "\n")

# conf matrices of "extremes", i.e. raw, added 20% of L1 and added 20% of L2
conf_raw_newseed3 = confusion_matrix(jm_raw, x_test, y_test)
conf_20_L1 = confusion_matrix(jm_add20_L1, x_test, y_test)
conf_20_L2 = confusion_matrix(jm_add20_L2, x_test, y_test)

# clean prints of the matrices
show(stdout, "text/plain", conf_raw)
show(stdout, "text/plain", conf_20_L1)
show(stdout, "text/plain", conf_20_L2)

function matrix_norm(A, B, norm)
	sum = 0
	for row in 1:10, col in 1:10
		sum += abs(A[row,col] - B[row,col])^norm
	end
	return sum^(1/norm)
end