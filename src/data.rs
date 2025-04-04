use {
    burn::{
        data::dataloader::{
            DataLoader, 
            DataLoaderBuilder
        },
        prelude::*
    },
    rand::Rng,
    std::sync::Arc,
};


pub struct SnakeBatch<B: Backend> {
    pub state: Tensor<B, 2>,
    pub rewards: Tensor<B, 2>,
}

pub struct SnakeBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> SnakeBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn load_data(&self, batch_size: usize, train: bool) -> Arc<dyn DataLoader<B>> {
        let mut batches: Vec<SnakeBatch<B>> = vec![];
    
        for _ in 0..batch_size {
            let state = Tensor::from_data::(vec![0.0; 100].as_slice(), &self.device); // Simuler spilltilstand
            let rewards = Tensor::from_data::<Vec<f32>>(vec![0.0; 4].as_slice(), &self.device); // Simuler belønninger
            batches.push(SnakeBatch { state, rewards });
        }
        
        let batches_into = batches.into();
        DataLoaderBuilder::new(batches_into)
            .batch_size(batch_size)
            .shuffle(if train { rand::thread_rng().gen_range(0..u64::MAX) } else { 0 })
            .build(batches_into)
    }
}
