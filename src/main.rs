mod training;
mod data;
mod model;
mod config;

use std::default;

use crate::{
    config::{TrainingConfig,ModelConfig},
};
use burn::optim::AdamConfig;
use burn::tensor::backend::AutodiffBackend;

fn main() {
    let config = TrainingConfig {
        model: ModelConfig {
            input_size: 100, // 10x10 rutenett
            hidden_size: 128,
            output_size: 4,  // Fire handlinger
        },
        optimizer: AdamConfig::new(),
        num_epochs: 50,
        batch_size: 32,
        num_workers: 4,
        seed: 42,
        learning_rate: 1e-3,
    };

    let device = AutodiffBackend::<f32>::default().device();
    training::train("artifacts", config, device);
}
