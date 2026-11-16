# Default Configs

These YAML configs represent TorchSig benchmark dataset configurations for
training and evaluating a model with independent train, validation, and test
datasets. Each split uses a distinct deterministic seed. Distinct seeds make
generation reproducible but do not guarantee that independently generated
datasets contain no overlapping examples.

| Name | YAML |
| ---- | ---- |
| Narrowband: Clean Train Dataset (all signals) | `narrowband_clean_train_all.yaml` |
| Narrowband: Clean Validation Dataset (all signals) | `narrowband_clean_val_all.yaml` |
| Narrowband: Clean Test Dataset (all signals) | `narrowband_clean_test_all.yaml` |
| Narrowband: Impaired Train Dataset (all signals) | `narrowband_impaired_train_all.yaml` |
| Narrowband: Impaired Validation Dataset (all signals) | `narrowband_impaired_val_all.yaml` |
| Narrowband: Impaired Test Dataset (all signals) | `narrowband_impaired_test_all.yaml` |
| Wideband: Clean Train Dataset (all signals) | `wideband_clean_train_all.yaml` |
| Wideband: Clean Validation Dataset (all signals) | `wideband_clean_val_all.yaml` |
| Wideband: Clean Test Dataset (all signals) | `wideband_clean_test_all.yaml` |
| Wideband: Impaired Train Dataset (all signals) | `wideband_impaired_train_all.yaml` |
| Wideband: Impaired Validation Dataset (all signals) | `wideband_impaired_val_all.yaml` |
| Wideband: Impaired Test Dataset (all signals) | `wideband_impaired_test_all.yaml` |
