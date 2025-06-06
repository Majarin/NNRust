use anyhow::Result;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Data, Tensor};
use burn_autodiff::ADBackendDecorator;
use burn_ndarray::NdArrayBackend;
use ndarray::prelude::*;
use ndarray_npy::NpzReader;
use image::io::Reader as ImageReader;
use std::fs::File;
use std::path::Path;
use clap::Parser;

type MyBackend = ADBackendDecorator<NdArrayBackend<f32>>;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the saved model
    #[arg(short, long)]
    model_path: String,
    
    /// Path to the image to predict
    #[arg(short, long)]
    image_path: String,
}

#[derive(Module, Debug)]
struct CNN<B: Backend> {
    conv1: burn::nn::conv::Conv2d<B>,
    conv2: burn::nn::conv::Conv2d<B>,
    fc1: burn::nn::Linear<B>,
    fc2: burn::nn::Linear<B>,
}

impl<B: Backend> CNN<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let batch_size = x.shape()[0];
        
        // First Conv -> ReLU -> MaxPool
        let x = self.conv1.forward(x);
        let x = x.relu();
        let x = x.max_pool2d([2, 2], [2, 2], [0, 0, 0, 0]);
        
        // Second Conv -> ReLU -> MaxPool
        let x = self.conv2.forward(x);
        let x = x.relu();
        let x = x.max_pool2d([2, 2], [2, 2], [0, 0, 0, 0]);
        
        // Flatten
        let x = x.reshape([batch_size, 7 * 7 * 64]);
        
        // Fully connected layers
        let x = self.fc1.forward(x);
        let x = x.relu();
        
        // Output layer
        self.fc2.forward(x)
    }
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();
    
    // Load the model
    let model_path = Path::new(&args.model_path);
    let model = CNN::<MyBackend>::load(model_path)?;
    
    // Load and preprocess the image
    let img = ImageReader::open(&args.image_path)?
        .decode()?
        .to_luma8();
    
    // Resize to 28x28 if needed
    let img = image::imageops::resize(
        &img,
        28,
        28,
        image::imageops::FilterType::Lanczos3,
    );
    
    // Convert to array and normalize
    let mut img_array = Array3::<f32>::zeros((1, 28, 28));
    for (i, j) in (0..28).flat_map(|i| (0..28).map(move |j| (i, j))) {
        img_array[[0, i, j]] = img.get_pixel(j as u32, i as u32)[0] as f32 / 255.0;
    }
    
    // Create tensor
    let img_tensor = Tensor::<MyBackend, 4>::from_data(
        Data::from(img_array.into_shape((1, 1, 28, 28)).unwrap())
    );
    
    // Make prediction
    let output = model.forward(img_tensor);
    let prediction = output.argmax(1).into_scalar().value();
    
    println!("Predicted class: {}", prediction);
    
    Ok(())
}