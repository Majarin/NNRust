use burn::{
    module::Param,
    nn::Linear,
    prelude::*,
}; // Bruker prelude for relevante moduler
use crate::config::ModelConfig; // Importerer ModelConfig fra config.rs

#[derive(Module, Debug, Clone)]
pub struct SnakeModel<B: Backend> {
    layer1: Param<Linear<B>>, // Første lineære lag
    layer2: Param<Linear<B>>, // Andre lineære lag
}

impl<B: Backend> SnakeModel<B> {
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            layer1: Linear::new(config.input_size, config.hidden_size),
            layer2: Linear::new(config.hidden_size, config.output_size),
        }
    }

    pub fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let hidden = self.layer1.forward(state).relu(); // ReLU aktivering
        self.layer2.forward(hidden)
    }
}
