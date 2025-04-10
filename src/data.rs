use {
    burn::{
        data::{
            dataloader::{
                batcher::Batcher,
                DataLoader,
                DataLoaderBuilder
            }, 
            dataset::InMemDataset
        },
        prelude::*
    },
    rand::Rng,
    std::sync::Arc,
};

#[derive(Clone, Debug)]
pub struct SnakeBatch<B: Backend> {
    pub state: Tensor<B, 2>,
    pub rewards: Tensor<B, 1, Int>,
}

#[derive(Clone, Debug)]
pub struct MyBatcher<B: Backend> {
    pub batches: Vec<SnakeBatch<B>>,
}

impl<B: Backend> Batcher<SnakeBatch<B>, Vec<SnakeBatch<B>>> for MyBatcher<B> {
    fn batch(&self, items: Vec<SnakeBatch<B>>) -> Vec<SnakeBatch<B>> {
        items
    }
}

pub struct SnakeBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> SnakeBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn load_data(&self, batch_size: usize, train: bool) -> Arc<dyn DataLoader<Vec<SnakeBatch<B>>>> {
        let mut batches: Vec<SnakeBatch<B>> = vec![];
    
        for _ in 0..batch_size {
            let state = Tensor::from_data(TensorData::from([0.0; 100]), &self.device); // Simuler spilltilstand
            let rewards = Tensor::from_data(TensorData::from([0.0; 4]), &self.device); // Simuler belønninger
            batches.push(SnakeBatch { state, rewards });
        }
        
        let build_dataset = InMemDataset::new(batches.clone());
        let new_batches = MyBatcher { batches };
        
        let data_loder: Arc<dyn DataLoader<Vec<SnakeBatch<B>>>> = DataLoaderBuilder::new(new_batches)
            .batch_size(batch_size)
            .shuffle(if train { rand::thread_rng().gen_range(0..u64::MAX) } else { 0 })
            .build(build_dataset);

        data_loder
    }
}