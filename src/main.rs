mod data;
mod config;
mod model;
mod training;

use crate::{
    config::{TrainingConfig, ModelConfig},
    data::SnakeBatcher
}; // Correcting the import to use config.rs
use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(
            ModelConfig::new(10, 512, 10), 
            AdamConfig::new()
        ),
        device.clone(),
    );
}
