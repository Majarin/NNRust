use burn::prelude::*; // Bruk prelude for Config
use burn::optim::AdamConfig;

#[derive(Config)]
pub struct ModelConfig {
    pub input_size: usize,  // Størrelse på input (spilltilstand)
    pub hidden_size: usize, // Antall noder i skjult lag
    pub output_size: usize, // Antall mulige handlinger
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,         // Konfigurasjon for modellen
    pub optimizer: AdamConfig,      // Konfigurasjon for optimizer
    #[config(default = 50)]
    pub num_epochs: usize,          // Antall trenings-epocher
    #[config(default = 32)]
    pub batch_size: usize,          // Batch-størrelse
    #[config(default = 4)]
    pub num_workers: usize,         // Antall parallelle arbeidertråder
    #[config(default = 42)]
    pub seed: u64,                  // Frø for reproduserbarhet
    #[config(default = 1.0e-3)]
    pub learning_rate: f64,         // Læringsrate
}
