use burn::{
    nn::{Linear, LinearConfig},
    tensor::activation::relu,
    prelude::*,
}; // Bruker prelude for relevante moduler
use crate::config::ModelConfig; // Importerer ModelConfig fra config.rs

#[derive(Module, Debug)]
pub struct SnakeModel<B: Backend> {
    layer1: Linear<B>, // Første lineære lag
    layer2: Linear<B>, // Andre lineære lag
}

impl<B: Backend> SnakeModel<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let layer1_config = LinearConfig::new(config.input_size, config.hidden_size); // Konfigurerer første lag
        let layer2_config = LinearConfig::new(config.hidden_size, config.output_size); // Konfigurerer andre lag

        Self {
            layer1: layer1_config.init(device), // Initialiserer første lag
            layer2: layer2_config.init(device), // Initialiserer andre lag
        }
    }

    pub fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let hidden = relu(self.layer1.forward(state)); // ReLU aktivering
        self.layer2.forward(hidden)
    }
}