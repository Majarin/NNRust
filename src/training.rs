use crate::{
    config::TrainingConfig,       // Importerer TrainingConfig fra config.rs
    data::{SnakeBatch, SnakeBatcher, MyBatcher},
    model::SnakeModel,
};
use burn::{
    data::dataloader::{batcher::Batcher, DataLoaderBuilder}, nn::loss::CrossEntropyLossConfig, prelude::*, record::CompactRecorder, tensor::backend::AutodiffBackend, train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
    }
};

impl<B: Backend> SnakeModel<B> {
    pub fn forward_classification(
        &self,
        state: Tensor<B, 2>,
        rewards: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        //let reshaped_state = state.flatten(1, usize::MAX);
        let output: Tensor<B, 2> = self.forward(state);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), rewards.clone());

        ClassificationOutput::new(loss, output, rewards)
    }
}

impl<B: AutodiffBackend> TrainStep<SnakeBatch<B>, ClassificationOutput<B>> for SnakeModel<B> {
    fn step(&self, batch: SnakeBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.state, batch.rewards);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<SnakeBatch<B>, ClassificationOutput<B>> for SnakeModel<B> {
    fn step(&self, batch: SnakeBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.state, batch.rewards)
    }
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

#[derive(Clone, Debug)]
pub struct CustomSnakeBatcher<B: Backend> {
    pub batches: SnakeBatcher<B>,
}

impl<B: Backend> Batcher<SnakeBatch<B>, Vec<SnakeBatch<B>>> for MyBatcher<B> {
    fn batch(&self, items: Vec<SnakeBatch<B>>) -> Vec<SnakeBatch<B>> {
        items
    }
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device, batches: Vec<SnakeBatch<B>>) -> Learner<SnakeModel<B>, B> {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train_not_dyn = SnakeBatcher::<B>::new(device.clone());
    let batcher_train = CustomSnakeBatcher { batches: batcher_train_not_dyn };
    let batcher_valid = SnakeBatcher::<B::InnerBackend>::new(device.clone());
    let dataset = InMemDataset::new(batches.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(batcher_train);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
