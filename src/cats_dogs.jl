
# NOTE! The dataset is the one from https://www.kaggle.com/datasets/tongpython/cat-and-dog?resource=download
# after duplicate images (i.e. all jpg.files having (1) in the title) have been removed

# train cats: 1:4000
# train dogs: 1:4000
# test  cats: 4001:5000
# test  dogs: 4001:5000

# binarizing algorithms:
# AdaptiveThreshold, Niblack, Polysegment, Sauvola

# binarizing algorithms that utilize single histogram-AdaptiveThreshold
# Otsu, Intermodes, MinimumError, Moments, UnimodalRosin, Entropy, Balanced, Yen

test_img = load("cats_and_dogs/test_set/test_set/cats/cat.4001.jpg")
test_gray_img = Gray.(test_img)
test_dim = (64,64)
test_sqr_img = imresize(test_gray_img, test_dim)
test_alg = Polysegment()
test_bin_img = binarize(test_sqr_img, test_alg)
test_pixel_values = channelview(test_sqr_img)

train_path = "cats_and_dogs/training_set/training_set"
test_path  = "cats_and_dogs/test_set/test_set"

function create_datasets(path)

	X = [] # features, i.e., rescaled grayscale images, stored as X[:,:,i]
	y = [] # labels (1,0) = (cats,dogs)
	dim = 64 # images will be resized to dim * dim pixels

	label_paths = ["cats", "dogs"]
	for animals in label_paths
		@assert animals == "cats" || animals == "dogs" "False label path"
		for label in readdir("$path/$animals")
			if label != "_DS_Store"
				img = load("$path/$animals/$label")
				gray_img = Gray.(img)
				sqr_img = imresize(gray_img, (dim, dim))
				pixel_values = reshape(Float32.(channelview(sqr_img)), (dim, dim, 1))

				if length(X) == 0 # if else to initialize the array X correctly
					X = pixel_values
				else
					X = cat(X, pixel_values, dims=3)
				end

				if animals == "cats"
					y = cat(y, 1, dims=1)
				elseif animals == "dogs"
					y = cat(y, 0, dims=1)
				end
			end
		end
	end
	return X, y
end

x_train_animal, y_train_animal = create_datasets(train_path) # NOTE! long runtime
x_test_animal, y_test_animal   = create_datasets(test_path)

y_train_onehot = onehotbatch(y_train_animal, 0:1)

# model without softmax to be in line with the JuMP model
m = Chain(
	Dense(64*64, 32, relu),
	Dense(32, 16, relu),
	Dense(16, 2)
)
m_len = length(m)

loss(x, y) = Flux.Losses.logitcrossentropy(m(x), y)
parameters = params(m)

x_train_flatten = flatten(x_train_animal)
x_test_flatten = flatten(x_test_animal)
train_data = [(x_train_flatten, y_train_onehot)]
test_data = [(x_test_flatten, y_test_animal)]

opt = ADAM(0.0001)

println("Value of the loss function at even steps")

n = 300
loss_values = zeros(n)
for i in 1:n
    train!(loss, parameters, train_data, opt)
	loss_values[i] = loss(x_train_flatten, y_train_onehot)
	if i % 50 == 0
		println(loss_values[i])
	end
end

correct_guesses = 0
test_len = length(y_test_animal)
for i in 1:test_len
    if findmax(m(test_data[1][1][:, i]))[2] - 1 == y_test_animal[i] # -1 to get right index
        correct_guesses += 1
    end
end

println("Accuracy: ", correct_guesses / test_len)
