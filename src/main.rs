use anyhow::Result;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Data, Tensor};
use burn_autodiff::ADBackendDecorator;
use burn_ndarray::NdArrayBackend;
use burn_train::{Config, TrainStep, ValidStep};
use burn_train::metric::{AccuracyMetric, CategoricalAccuracy};
use burn_train::metric::store::{Aggregate, Store};
use burn_train::data::{dataloader, dataset::Dataset};
use ndarray::prelude::*;
use ndarray_npy::NpzReader;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;
use clap::Parser;

// Vi bruker NdArrayBackend istedenfor TchBackend
type MyBackend = ADBackendDecorator<NdArrayBackend<f32>>;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the .npz file containing training data
    #[arg(short, long)]
    npz_file: String,
    
    /// Number of training epochs
    #[arg(short, long, default_value_t = 10)]
    epochs: usize,
    
    /// Batch size for training
    #[arg(short, long, default_value_t = 32)]
    batch_size: usize,
    
    /// Learning rate
    #[arg(short, long, default_value_t = 0.001)]
    learning_rate: f32,
}

// Data structures for our datasets
#[derive(Clone, Debug)]
struct ImageData {
    images: Array4<f32>, // [N, C, H, W]
    labels: Array2<u32>, // [N, 1]
}

// Our dataset implementation
struct ImageDataset {
    data: ImageData,
}

impl ImageDataset {
    fn new(data: ImageData) -> Self {
        Self { data }
    }
    
    // Create a dataset from an .npz file
    fn from_npz(file_path: &str) -> Result<(Self, Self)> {
        let file = File::open(file_path)?;
        let mut npz = NpzReader::new(file)?;
        
        // Load images and labels from npz file
        let all_images: Array4<f32> = npz.by_name("images")?;
        let all_labels: Array2<u32> = npz.by_name("labels")?;
        
        // Normalize images to [0, 1] range
        let all_images = all_images / 255.0;
        
        // Get total number of examples
        let n_samples = all_images.shape()[0];
        let n_train = (n_samples as f32 * 0.8) as usize; // 80% training, 20% validation
        
        // Split data
        let train_images = all_images.slice(s![0..n_train, .., .., ..]).to_owned();
        let train_labels = all_labels.slice(s![0..n_train, ..]).to_owned();
        
        let valid_images = all_images.slice(s![n_train.., .., .., ..]).to_owned();
        let valid_labels = all_labels.slice(s![n_train.., ..]).to_owned();
        
        let train_data = ImageData { images: train_images, labels: train_labels };
        let valid_data = ImageData { images: valid_images, labels: valid_labels };
        
        Ok((Self::new(train_data), Self::new(valid_data)))
    }
}

impl Dataset for ImageDataset {
    type Item = (Tensor<MyBackend, 3>, Tensor<MyBackend, 1>);

    fn get(&self, index: usize) -> Self::Item {
        let image = self.data.images.slice(s![index, .., .., ..]).to_owned();
        let label = self.data.labels.slice(s![index, ..]).to_owned();
        
        // Convert to tensors
        let image_tensor = Tensor::<MyBackend, 3>::from_data(
            Data::from(image.into_shape((image.shape()[0], image.shape()[1], image.shape()[2])).unwrap())
        );
        
        let label_tensor = Tensor::<MyBackend, 1>::from_data(
            Data::from(label.into_shape(label.shape()[0]).unwrap())
        );
        
        (image_tensor, label_tensor)
    }

    fn len(&self) -> usize {
        self.data.images.shape()[0]
    }
}

// Define our training batch type
#[derive(Clone, Debug)]
struct Batch {
    images: Tensor<MyBackend, 4>, // [B, C, H, W]
    targets: Tensor<MyBackend, 1>, // [B]
}

// Implementation of the CNN model
#[derive(Module, Debug)]
struct CNN<B: Backend> {
    conv1: burn::nn::conv::Conv2d<B>,
    conv2: burn::nn::conv::Conv2d<B>,
    fc1: burn::nn::Linear<B>,
    fc2: burn::nn::Linear<B>,
}

impl<B: Backend> CNN<B> {
    fn new(num_classes: usize) -> Self {
        let device = B::Device::default();
        
        Self {
            conv1: burn::nn::conv::Conv2dConfig::new([1, 32], [3, 3])
                .with_stride([1, 1])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(&device),
            conv2: burn::nn::conv::Conv2dConfig::new([32, 64], [3, 3])
                .with_stride([1, 1])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(&device),
            fc1: burn::nn::LinearConfig::new(7 * 7 * 64, 128).init(&device),
            fc2: burn::nn::LinearConfig::new(128, num_classes).init(&device),
        }
    }
    
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

// Training configuration
#[derive(Serialize, Deserialize, Debug, Clone)]
struct TrainingConfig {
    learning_rate: f32,
    num_epochs: usize,
}

impl Config for TrainingConfig {
    type Module = CNN<MyBackend>;
    type Metric = AccuracyMetric<usize>;
    type Optimizers = burn_train::optim::AdamOptimizers<Self::Module>;
}

struct ImageRecognitionTrainStep;

impl TrainStep<Batch, TrainingConfig> for ImageRecognitionTrainStep {
    fn step(
        state: &mut burn_train::TrainState<TrainingConfig>,
        batch: Batch,
    ) -> burn_train::StepResult<TrainingConfig> {
        let Batch { images, targets } = batch;
        
        // Forward pass
        let output = state.model.forward(images);
        
        // Calculate loss
        let targets = targets.float();
        let loss = output.cross_entropy_with_logits_loss(targets);
        
        // Backward pass
        let gradients = loss.backward();
        
        // Optimization step
        state.optimizers.adam_step(&gradients);
        
        // Get predictions for metrics
        let predictions = output.argmax(1);
        let targets_usize = targets.map(|x| x as usize);
        
        // Store metrics
        state.metrics
            .loss
            .update(loss.into_scalar().value());
        
        state.metrics
            .store
            .iter_mut()
            .for_each(|metrics| metrics.update(&predictions, &targets_usize));
        
        Ok(())
    }
}

struct ImageRecognitionValidStep;

impl ValidStep<Batch, TrainingConfig> for ImageRecognitionValidStep {
    fn step(
        state: &mut burn_train::ValidState<TrainingConfig>,
        batch: Batch,
    ) -> burn_train::StepResult<TrainingConfig> {
        let Batch { images, targets } = batch;
        
        // Forward pass
        let output = state.model.forward(images);
        
        // Calculate loss
        let targets = targets.float();
        let loss = output.cross_entropy_with_logits_loss(targets);
        
        // Get predictions for metrics
        let predictions = output.argmax(1);
        let targets_usize = targets.map(|x| x as usize);
        
        // Store metrics
        state.metrics
            .loss
            .update(loss.into_scalar().value());
        
        state.metrics
            .store
            .iter_mut()
            .for_each(|metrics| metrics.update(&predictions, &targets_usize));
        
        Ok(())
    }
}

fn collate_fn(items: Vec<(Tensor<MyBackend, 3>, Tensor<MyBackend, 1>)>) -> Batch {
    let batch_size = items.len();
    
    // Extract images and targets
    let (images, targets): (Vec<_>, Vec<_>) = items.into_iter().unzip();
    
    // Stack tensors
    let images = Tensor::stack(images, 0);
    let targets = Tensor::cat(targets, 0);
    
    Batch { images, targets }
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();
    
    // Load data from npz file
    let (train_dataset, valid_dataset) = ImageDataset::from_npz(&args.npz_file)?;
    
    println!("Loaded {} training samples and {} validation samples.", 
             train_dataset.len(), valid_dataset.len());
    
    // Create data loaders
    let train_loader = dataloader::DataLoaderBuilder::new(train_dataset)
        .batch_size(args.batch_size)
        .shuffle(true)
        .num_workers(2)
        .build(collate_fn);
    
    let valid_loader = dataloader::DataLoaderBuilder::new(valid_dataset)
        .batch_size(args.batch_size)
        .shuffle(false)
        .num_workers(2)
        .build(collate_fn);
    
    // Infer the number of classes from the data
    let num_classes = 10; // Assuming 10 classes, like MNIST
    
    // Initialize model
    let model = CNN::<MyBackend>::new(num_classes);
    
    // Create optimizer
    let optimizers = burn_train::optim::AdamOptimizers::new(
        model.clone(),
        burn_train::optim::AdamConfig::new().with_learning_rate(args.learning_rate),
    );
    
    // Setup metrics
    let metrics_store = Store::new(
        vec![CategoricalAccuracy::new()],
        vec![Aggregate::Average],
    );
    
    // Setup config
    let config = TrainingConfig {
        learning_rate: args.learning_rate,
        num_epochs: args.epochs,
    };
    
    // Setup model state
    let mut state = burn_train::TrainState::new(
        model,
        optimizers,
        config.clone(),
        metrics_store,
    );
    
    // Training loop
    for epoch in 0..config.num_epochs {
        println!("Epoch {}/{}", epoch + 1, config.num_epochs);
        
        // Training
        let train_metrics = burn_train::train_epoch(
            &mut state,
            &train_loader,
            ImageRecognitionTrainStep,
        )?;
        
        // Validation
        let valid_metrics = burn_train::valid_epoch(
            &mut state.valid_state(),
            &valid_loader,
            ImageRecognitionValidStep,
        )?;
        
        println!(
            "Train Loss: {:.4}, Train Accuracy: {:.4}, Valid Loss: {:.4}, Valid Accuracy: {:.4}",
            train_metrics.loss,
            train_metrics.store.values()[0],
            valid_metrics.loss,
            valid_metrics.store.values()[0],
        );
    }
    
    // Save model
    let save_path = Path::new("model.bin");
    state.model.save(save_path)?;
    println!("Model saved to {:?}", save_path);
    
    Ok(())
}