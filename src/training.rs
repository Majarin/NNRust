use crate::{
    config::TrainingConfig,       // Importerer TrainingConfig fra config.rs
    data::{SnakeBatch, SnakeBatcher},
    model::SnakeModel,
};
use burn::{
    data::dataloader::DataLoaderBuilder,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
    },
    nn::loss::CrossEntropyLossConfig,
};

impl<B: Backend> SnakeModel<B> {
    pub fn forward_classification(
        &self,
        state: Tensor<B, 3>,
        rewards: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(state);
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

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = SnakeBatcher::<B>::new(device.clone());
    let batcher_valid = SnakeBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build();

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build();

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
