use {
    burn::{
        data::dataloader::{
            DataLoader, 
            DataLoaderBuilder
        },
        data::dataset::InMemDataset,
        prelude::*
    },
    rand::Rng,
    std::sync::Arc,
};

#[derive(Clone, Debug)]
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
            let state = Tensor::from_data(TensorData::from([0.0; 100]), &self.device); // Simuler spilltilstand
            let rewards = Tensor::from_data(TensorData::from([0.0; 4]), &self.device); // Simuler belønninger
            batches.push(SnakeBatch { state, rewards });
        }
        
        let dataset = InMemDataset::new(batches);
        
        DataLoaderBuilder::new(batches)
            .batch_size(batch_size)
            .shuffle(if train { rand::thread_rng().gen_range(0..u64::MAX) } else { 0 })
            .build(dataset)
    }
}
