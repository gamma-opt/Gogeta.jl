
function create_sum_nn(train_len, test_len, summand_count, model) # a neural network that solves sum of m variables

	x_train      = rand(train_len, summand_count) # pairs of random float32's in the range of [0,1]
	y_train      = [sum(x_train[i,:]) for i in 1:train_len] # sums of these pairs for training
	x_test       = round.(rand(test_len, summand_count))
	y_test       = [sum(x_test[i,:]) for i in 1:test_len]

	loss(x, y) = mse(model(x), y)
	parameters = params(model)
	dataset = [(x_train', y_train')]
	opt = ADAM(0.01)

	println("Value of the loss function at even steps")
	n = 100
	loss_values = zeros(n)
	for i in 1:n
		train!(loss, parameters, dataset, opt)
		loss_values[i] = loss(x_train', y_train')
		if i % 20 == 0
			println(loss_values[i])
		end
	end

	println("\nPrediction of the data and the expected output:")
	println(model(x_test'))
	println(y_test')

	return model
end

function create_rosenbrock_nn(train_len, test_len, model) # solves rosenbrock(x,y) with a = b = 1

	a = 1
	b = 1
	rosenbrock(x,y) = (a-x)^2 + b*(y-x^2)^2

	x_train      = rand(train_len, 2) # pairs of random float32's in the range of [0,1]
	y_train      = [rosenbrock(x_train[i], x_train[i+train_len]) for i in 1:train_len]
	x_test       = round.(rand(test_len, 2)) 
	y_test       = [rosenbrock(x_test[i], x_test[i+test_len]) for i in 1:test_len]

	loss(x, y) = mse(model(x), y)
	parameters = params(model)
	dataset = [(x_train', y_train')]
	opt = ADAM(0.001)

	println("Value of the loss function at even steps")
	n = 200
	loss_values = zeros(n)
	for i in 1:n
		train!(loss, parameters, dataset, opt)
		loss_values[i] = loss(x_train', y_train')
		if i % 40 == 0.0
			println(loss_values[i])
		end
	end

	println("\nPrediction of the data and the expected output:")
	println(model(x_test'))
	println(y_test')

	return model
end
