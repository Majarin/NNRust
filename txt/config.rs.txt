use burn::prelude::*; // Bruk prelude for Config
use burn::optim::AdamConfig;

#[derive(Config)]
pub struct ModelConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 50)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-3)]
    pub learning_rate: f64,
}
