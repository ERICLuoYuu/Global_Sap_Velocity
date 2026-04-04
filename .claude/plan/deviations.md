# Deviations Log

## Deviation 1: 25 features, not 24
- **Phase**: Phase 3 (Implementation)
- **Planned**: plan.md and paper_extraction.md stated 24 input features
- **Actual**: 25 features — the `time_hours` column (cyclical cosine encoding of hour: `-cos(2pi*hour/24)`) was missed in the initial extraction. It is created in Cell 6 of the gauged notebook and included in the feature set via the `unique_column` collection in Cell 7-8.
- **Reason**: The cyclical_encode function (Cell 4) and its usage were documented but its inclusion in the final feature count was not tracked through the dynamic column discovery logic.
- **Impact on acceptance criteria**: `input_size = len(features)` will be 25 at runtime, not 24. This is faithful to their code.
- **Approved**: yes (auto — matches their code exactly)

## Deviation 2: Replaced dynamic variable creation with dict
- **Phase**: Phase 3 (Implementation)
- **Planned**: plan.md said "faithful port" of their code
- **Actual**: Replaced all `exec('{} = df_oneyear'.format(single_name))` patterns with `all_data[plant_year_id] = df_oneyear` dict storage. Also replaced `exec("plant = pd.DataFrame({}, columns=all_columns)".format(plant_id))` with `pd.DataFrame(all_data[plant_id], columns=all_columns)`.
- **Reason**: Dynamic variable creation via exec() is a known anti-pattern. Dict-based storage is functionally identical but safer and testable.
- **Impact on acceptance criteria**: None — behavior is identical, only storage mechanism differs.
- **Approved**: yes (auto — trivial refactor, no behavioral change)

## Deviation 3: Gradient clipping placement
- **Phase**: Phase 3 (Implementation)
- **Planned**: Standard placement (clip before step)
- **Actual**: Gradient clipping placed AFTER optimizer.step(), matching their Cell 18 code exactly. This is unusual but we replicate it faithfully.
- **Reason**: Their code calls `optimizer.step()` then `clip_grad_norm_()`. This means clipping affects the next iteration's gradients, not the current step's update. We replicate this quirk for fidelity.
- **Impact on acceptance criteria**: None — matches their code.
- **Approved**: yes (auto — faithful replication)

## Deviation 4: network.eval() avoided in trainer
- **Phase**: Phase 3 (Implementation)
- **Planned**: Standard practice would use network.eval() in train_model
- **Actual**: Their code does NOT call network.eval() during validation within the training loop. Dropout remains active during validation loss computation. We replicate this.
- **Reason**: Faithful to their Cell 18 which only calls network.eval() after the full training loop completes.
- **Impact on acceptance criteria**: Validation loss numbers will include dropout effects, but this matches their behavior.
- **Approved**: yes (auto — faithful replication)
