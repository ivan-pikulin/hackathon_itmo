from torch import nn
from torch.optim import Adam

from Source.models import FCNN
from Source.preprocessing import get_example_dataset, train_test_valid_split
from Source.trainer import ModelTrainer

n_split = 5
batch_size = 64
output_path = "Output"
output_mark = f"Test"

dataset = get_example_dataset()
folds, test_dl = train_test_valid_split(dataset, n_splits=5, test_ratio=0.2, batch_size=64)

model = FCNN(
    batch_size=batch_size,
    n_features=4,
    n_targets=1,
    hidden_dims=[64],
    dropout=0,
    bn=True,
    actf=nn.LeakyReLU(),
    optimizer=Adam,
    optimizer_parameters=None,
)

trainer = ModelTrainer(
    model=model,
    train_valid_data=folds,
    test_data=test_dl,
    output_folder=output_path,
    out_folder_mark=output_mark,
    es_patience=100,
    epochs=1000,
    verbose=True
)

trainer.train_cv_models()
