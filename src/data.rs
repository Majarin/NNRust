use burn::prelude::*;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};

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

    pub fn load_data(&self, batch_size: usize, train: bool) -> DataLoader<B, SnakeBatch<B>> {
        let mut batches = vec![];

        for _ in 0..batch_size {
            let state = Tensor::from_data(vec![0.0; 100]); // Simuler spilltilstand
            let rewards = Tensor::from_data(vec![0.0; 4]); // Simuler belønninger
            batches.push(SnakeBatch { state, rewards });
        }

        DataLoaderBuilder::new(batches)
            .batch_size(batch_size)
            .shuffle(train)
            .build()
    }
}
