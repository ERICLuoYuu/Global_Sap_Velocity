"""Turning flags into a mask -- one algorithm, three mechanisms.

Deciding what to mask and applying the mask are separate steps.  That is the
whole reason this module returns a boolean frame instead of a cleaned one: the
decision can be counted, reported and compared against an independent census
*before* anything is destroyed.  The old inline code did both at once, so the
only record of what it had removed was a print statement.
"""

from __future__ import annotations

import pandas as pd

from src.qc_flags.columns import qc_column_for, resolve_flag_column, split_qc_columns
from src.qc_flags.config import QcFlagConfig
from src.qc_flags.mechanisms import Mechanism
from src.qc_flags.vocabulary import actionable_mask


def _deduplicate_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.index.duplicated().any():
        return frame[~frame.index.duplicated(keep="first")]
    return frame


def qc_mask_for_column(values: pd.Series, threshold: int) -> pd.Series:
    """Mask where the QC level is at or above ``threshold``.

    Non-numeric and missing verdicts are masked.  An absent QC value is not
    evidence of quality, and the conservative direction here costs a handful of
    rows while the permissive one would let unlabelled data into training.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.isna() | (numeric >= threshold)


def _sapfluxnet_mask(
    data: pd.DataFrame,
    flags: pd.DataFrame,
    variables,
    config: QcFlagConfig,
) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=data.index, columns=variables)
    flags = _deduplicate_index(flags)
    for variable in variables:
        flag_column = resolve_flag_column(variable, flags.columns)
        if flag_column is None:
            continue
        # Align on the timestamp index, never on position: a flag frame sorted
        # differently from its data frame would otherwise mask the wrong hours.
        aligned = flags[flag_column].reindex(data.index)
        mask[variable] = actionable_mask(aligned, config.actionable_tokens)
    return mask


def _qc_column_mask(data: pd.DataFrame, variables, config: QcFlagConfig) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=data.index, columns=variables)
    for variable in variables:
        qc_name = qc_column_for(variable)
        if qc_name in data.columns:
            mask[variable] = qc_mask_for_column(data[qc_name], config.qc_threshold)
    return mask


def propagate_to_derived(mask: pd.DataFrame, config: QcFlagConfig) -> pd.DataFrame:
    """Union each derived variable's mask with its sources'.

    Union, not replacement: if a derived variable ever gains quality
    information of its own, propagation must add to it rather than overwrite
    it.
    """
    out = mask.copy()
    for derived, sources in config.derived_from.items():
        if derived not in out.columns:
            continue
        combined = out[derived]
        for source in sources:
            if source in out.columns:
                combined = combined | out[source]
        out[derived] = combined
    return out


def build_mask(
    data: pd.DataFrame,
    flags: pd.DataFrame | None,
    mechanism: Mechanism,
    config: QcFlagConfig | None = None,
) -> pd.DataFrame:
    """A boolean frame over ``data``'s variables: True means "set to NaN".

    QC columns are never themselves maskable variables (D-3), and every
    mechanism -- including :attr:`Mechanism.NONE` -- returns a frame of the
    right shape.  A site with no flags is an ordinary site with an all-False
    mask, not an error (D-1).
    """
    if config is None:
        config = QcFlagConfig()
    variables, _ = split_qc_columns(data)

    if mechanism is Mechanism.SAPFLUXNET_FLAGS and flags is not None:
        # NO propagation here. `ppfd_in` and `rh` are synthesised by
        # `icos_data_extractor` -- that is a fact about OUR extraction, not
        # about the archive. Measured over the corpus: 19 of 176 external sites
        # carry `ppfd_in` on rows where `sw_in` is absent (154,155 rows;
        # USA_DUK_HAR alone 36,047 of 70,128 present), and 35 of 200 do the same
        # for `rh`. A value present where its source is missing cannot have been
        # computed from that source, so at these sites both are independent
        # measurements with their own flag columns. Propagating would destroy
        # real observations -- and would break criterion 4 besides.
        return _sapfluxnet_mask(data, flags, variables, config)
    if mechanism is Mechanism.QC_COLUMNS:
        # Propagation applies here and only here: at all 19 ICOS sites the
        # derived variables have ZERO orphan rows, exactly as the extractor
        # implies. Masking `sw_in` while keeping `ppfd_in` would retain a number
        # computed from data we just rejected.
        return propagate_to_derived(_qc_column_mask(data, variables, config), config)
    return pd.DataFrame(False, index=data.index, columns=variables)


def apply_mask(data: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``data`` with the masked cells set to NaN.

    Never mutates its input. Columns absent from the mask -- the QC columns
    themselves, timestamps, coordinates -- are carried through untouched.
    """
    out = data.copy()
    for column in mask.columns:
        if column not in out.columns:
            continue
        selector = mask[column].reindex(out.index, fill_value=False).fillna(False).astype(bool)
        # A maskable column is float-valued, whether or not anything is masked
        # today. The old code got this by accident -- it assigned NaN through
        # `.loc` unconditionally, which upcast int64 to float64 even when the
        # mask was empty -- and the accident is load-bearing: `precip` at
        # CHN_YIN_ST3_DRO is integral, so skipping the upcast rewrote every
        # `0.0` as `0` and changed two delivered files without changing a
        # single value. Done explicitly here, both to keep those files
        # byte-identical and to avoid the `.loc`-assign-NaN path, which is
        # deprecated in pandas 2.x and raises in 3.0.
        if pd.api.types.is_integer_dtype(out[column]):
            out[column] = out[column].astype(float)
        if selector.any():
            out[column] = out[column].where(~selector)
    return out


def count_masked(mask: pd.DataFrame, data: pd.DataFrame):
    """Per variable: how many *present* values the mask removes.

    Masking an already-NaN cell is a no-op, so counting raw mask hits would
    overstate the loss. The denominator is the count of values that were
    actually there.
    """
    masked = {}
    present = {}
    for column in mask.columns:
        if column not in data.columns:
            continue
        is_present = data[column].notna()
        present[column] = int(is_present.sum())
        selector = mask[column].reindex(data.index, fill_value=False).fillna(False).astype(bool)
        masked[column] = int((selector & is_present).sum())
    return masked, present
