use burn::module::Module;
use burn::nn::{Linear, Relu, CrossEntropyLoss};
use burn::tensor::{backend::TchBackend, Tensor};
use burn::train::optimizer::Adam;
use ndarray_npy::NpzReader;
use std::fs::File;
use ndarray::Array3;

type Backend = TchBackend<f32>;

#[derive(Module, Debug)]
struct Model {
    fc1: Linear<Backend>,
    fc2: Linear<Backend>,
}

impl Model {
    fn new() -> Self {
        Self {
            fc1: Linear::new(28 * 28, 64),
            fc2: Linear::new(64, 10), // 10 former?
        }
    }

    fn forward(&self, input: Tensor<Backend, 2>) -> Tensor<Backend, 2> {
        let x = self.fc1.forward(input).Relu();
        self.fc2.forward(x)
    }
}

fn load_dataset(path: &str) -> (Tensor<Backend, 2>, Tensor<Backend, 1>) {
    let file = File::open(path).expect("Kunne ikke åpne .npz");
    let mut npz = NpzReader::new(file).unwrap();

    // Henter ut bildene (antatt float32 og form [N, 28, 28])
    let images: Array3<f32> = npz.by_name("images.npy").unwrap();
    let num_samples = images.len_of(ndarray::Axis(0));

    // Flatten hver 28x28 til 784
    let images_flat = images.into_shape((num_samples, 28 * 28)).unwrap();
    let images_vec = images_flat.iter().cloned().collect::<Vec<f32>>();
    let images_tensor: Tensor<usize, 2> = Tensor::<Backend, 2>::from_floats(images_vec, [num_samples, 28 * 28]);

    // Henter ut etikettene
    let labels: ndarray::Array1<u8> = npz.by_name("labels.npy").unwrap();
    let labels_vec = labels.iter().map(|x| *x as i64).collect::<Vec<i64>>();
    let labels_tensor = Tensor::<Backend, 1>::from_ints(labels_vec, [num_samples]);

    (images_tensor, labels_tensor)
}

fn main() {
    let (x_train, y_train) = load_dataset("shape_dataset.npz");

    let mut model = Model::new();
    let mut optimizer = Adam::new();

    for epoch in 0..20 {
        let logits = model.forward(x_train.clone());
        let loss = CrossEntropyLoss::new().forward(logits, y_train.clone());

        println!("Epoch {epoch}: loss = {:?}", loss.clone().into_scalar());

        let grads = loss.backward();
        optimizer.step(&mut model, &grads);
    }

    println!("Trening ferdig.");
}
