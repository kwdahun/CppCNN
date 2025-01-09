#include <iostream>
#include "Matrix.h"
#include "Layer.h"
#include "Linear.h"
#include "ReLU.h"
#include "Flatten.h"
#include "Sequential.h"
#include "CrossEntropyLoss.h"
#include "MNISTReader.h"

int main() {
	Sequential model = Sequential(Flatten())
		.add(Linear(784, 20))
		.add(ReLU())
		.add(Linear(20, 10));

	const std::string train_images = std::string(DATA_DIR) + "/train-images.idx3-ubyte";
	const std::string train_labels = std::string(DATA_DIR) + "/train-labels.idx1-ubyte";
	const size_t batch_size = 32;
	const float learning_rate = 0.01f;
	const size_t epochs = 10;

	for (size_t epoch = 0; epoch < epochs; epoch++) {
		float epoch_loss = 0.0f;
		size_t batches = 0;

		for (size_t start_idx = 0; start_idx < 60000; start_idx += batch_size) {
			Matrix images = MNISTReader::readImageBatch(train_images, batch_size, start_idx);
			Matrix labels = MNISTReader::readLabelBatch(train_labels, batch_size, start_idx);

			Matrix logits = model.forward(images);

			float loss = CrossEntropyLoss::compute(logits, labels);
			Matrix gradient = CrossEntropyLoss::compute_gradient(logits, labels);

			model.backward(gradient);
			model.update_parameters(learning_rate);

			epoch_loss += loss;
			batches++;

			if (batches % 100 == 0) {
				std::cout << "Epoch " << epoch << ", Batch " << batches
					<< ", Loss: " << epoch_loss / batches << std::endl;
			}
		}

		std::cout << "Epoch " << epoch << " completed. Average loss: "
			<< epoch_loss / batches << std::endl;
	}

	return 0;
}