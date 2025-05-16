use burn::module::Module;
use burn::nn::{LinearConfig, loss::CrossEntropyLoss};
use burn::tensor::Tensor;
use burn::backend::Autodiff;
use burn::optim::Adam;
use ndarray_npy::NpzReader;
use std::fs::File;
use ndarray::Array3;
use std::result::Result;

type Backend = Autodiff<f32>;

#[derive(Module, Debug, Clone)]
struct Model {
    fc1: Linear<Backend>,
    fc2: Linear<Backend>,
}

impl Model {
    fn new() -> Self {
        Self {
            fc1: Linear::new(28 * 28, 64).unwrap(),
            fc2: Linear::new(64, 10).unwrap(),
        }
    }

    fn forward(&self, input: &Tensor<Backend, 2>) -> Result<Tensor<Backend, 2>, String> {
        let x = self.fc1.forward(input).map_err(|e| e.to_string())?.relu();
        self.fc2.forward(&x).map_err(|e| e.to_string())
    }
}

fn load_dataset(path: &str) -> Result<(Tensor<Backend, 2>, Tensor<Backend, 1>), String> {
    let file = File::open(path).map_err(|e| format!("Could not open {}: {}", path, e))?;
    let mut npz = NpzReader::new(file).map_err(|e| format!("Could not read {}: {}", path, e))?;

    let images: Array3<f32> = npz.by_name("images.npy").map_err(|e| format!("Could not extract images: {}", e))?;
    let num_samples = images.len_of(ndarray::Axis(0));

    let images_flat = images.into_shape((num_samples, 28 * 28)).map_err(|e| format!("Could not reshape images: {}", e))?;
    let images_vec = images_flat.iter().cloned().collect::<Vec<f32>>();
    let images_tensor = Tensor::<Backend, 2>::from_floats(images_vec, [num_samples, 28 * 28]).map_err(|e| e.to_string())?;

    let labels: ndarray::Array1<u8> = npz.by_name("labels.npy").map_err(|e| format!("Could not extract labels: {}", e))?;
    let labels_vec = labels.iter().map(|x| *x as i64).collect::<Vec<i64>>();
    let labels_tensor = Tensor::<Backend, 1>::from_ints(labels_vec, [num_samples]).map_err(|e| e.to_string())?;

    Ok((images_tensor, labels_tensor))
}

fn main() {
    let (x_train, y_train) = match load_dataset("shape_dataset.npz") {
        Ok(x) => x,
        Err(e) => {
            eprintln!("Error: {}", e);
            return;
        }
    };

    let mut model = Model::new();
    let mut optimizer = Adam::new();

    for epoch in 0..20 {
        let logits = match model.forward(&x_train) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("Error in forward: {}", e);
                return;
            }
        };
        let loss = match CrossEntropyLoss::new().forward(logits, y_train.clone()) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("Error in CrossEntropyLoss: {}", e);
                return;
            }
        };

        println!("Epoch {}: loss = {:?}", epoch, loss.clone().into_scalar());

        let grads = match loss.backward() {
            Ok(x) => x,
            Err(e) => {
                eprintln!("Error in backward: {}", e);
                return;
            }
        };
        if let Err(e) = optimizer.step(&mut model, &grads) {
            eprintln!("Error in Adam: {}", e);
            return;
        }
    }

    println!("Training complete.");

    // Evaluér modellen på test-data
    let (x_test, y_test) = match load_dataset("shape_dataset.npz") {
        Ok(x) => x,
        Err(e) => {
            eprintln!("Error: {}", e);
            return;
        }
    };

    let test_logits = model.forward(&x_test);
    let test_loss = CrossEntropyLoss::new().forward(test_logits, y_test.clone());
    println!("Test loss = {:?}", test_loss.clone().into_scalar());

    let test_acc = accuracy(test_logits, y_test);
    println!("Test accuracy = {:?}", test_acc);
}

// Definer en accuracy-funksjon
fn accuracy(logits: Tensor<Backend, 2>, targets: Tensor<Backend, 1>) -> f64 {
    let predictions = logits.argmax(1);
    let correct = predictions.eq(targets);
    correct.mean().into_scalar()
}