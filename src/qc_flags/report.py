"""Every site ends in a recorded state.

D-1's real damage was not the ``AttributeError``.  It was that the error became
one line of stdout among thousands -- "Error loading X" -- and from the outside
a site that was deliberately skipped looked exactly like a site that died.  The
question "did all 226 sites get processed?" had no answer short of reading a
log.

Here it is a number.  A site is accounted for if it carries either measured
counts or an explicit skip reason; anything else is a site that vanished, and
:meth:`RunReport.unaccounted_sites` names it.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path

from src.qc_flags.mechanisms import Mechanism


@dataclasses.dataclass(frozen=True)
class SiteFlagOutcome:
    """What happened to one site, for one kind of file."""

    site: str
    kind: str
    mechanism: Mechanism
    masked_counts: Mapping[str, int]
    present_counts: Mapping[str, int]
    skipped_reason: str | None = None

    @property
    def key(self):
        return (self.site, self.kind)

    @property
    def is_accounted_for(self) -> bool:
        if self.skipped_reason:
            return True
        return bool(self.masked_counts) or bool(self.present_counts)

    def masked_fraction(self, variable: str) -> float:
        """Masked values as a fraction of the values that were present.

        Masking an already-NaN cell removes nothing, so the present count is
        the only honest denominator.
        """
        present = self.present_counts.get(variable, 0)
        if not present:
            return 0.0
        return self.masked_counts.get(variable, 0) / present

    def to_dict(self) -> dict:
        return {
            "site": self.site,
            "kind": self.kind,
            "mechanism": self.mechanism.value,
            "masked_counts": dict(self.masked_counts),
            "present_counts": dict(self.present_counts),
            "skipped_reason": self.skipped_reason,
        }


class RunReport:
    """The outcome of one QC pass over a corpus."""

    def __init__(self) -> None:
        self._outcomes: dict = {}

    def __len__(self) -> int:
        return len(self._outcomes)

    def add(self, outcome: SiteFlagOutcome) -> None:
        """Record one site.

        Raises:
            ValueError: if the same (site, kind) is added twice. Silent
                overwriting would let ``sites_in == sites_reported`` hold while
                a site was lost, which is precisely what criterion 2 is meant to
                detect.
        """
        if outcome.key in self._outcomes:
            raise ValueError(f"outcome already recorded for {outcome.site} ({outcome.kind})")
        self._outcomes[outcome.key] = outcome

    @property
    def outcomes(self) -> list[SiteFlagOutcome]:
        return list(self._outcomes.values())

    def mechanism_census(self) -> dict[str, int]:
        """How many sites of each mechanism, plus how many were skipped.

        A skipped site counts only as skipped: it never reached a mechanism, so
        filing it under one would inflate that population against criterion 1.
        """
        census = {m.value: 0 for m in Mechanism}
        census["skipped"] = 0
        for outcome in self._outcomes.values():
            if outcome.skipped_reason:
                census["skipped"] += 1
            else:
                census[outcome.mechanism.value] += 1
        return census

    def unaccounted_sites(self) -> list[str]:
        return sorted(o.site for o in self._outcomes.values() if not o.is_accounted_for)

    def all_accounted_for(self) -> bool:
        return not self.unaccounted_sites()

    def to_dict(self) -> dict:
        return {
            "census": self.mechanism_census(),
            "sites_reported": len(self._outcomes),
            "unaccounted_sites": self.unaccounted_sites(),
            "sites": [o.to_dict() for o in self._outcomes.values()],
        }

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def to_table(self) -> str:
        """A printable summary. Skip reasons are shown, never elided."""
        lines = ["site            kind  mechanism          masked / present"]
        for outcome in sorted(self._outcomes.values(), key=lambda o: o.key):
            if outcome.skipped_reason:
                detail = f"SKIPPED: {outcome.skipped_reason}"
            else:
                total_masked = sum(outcome.masked_counts.values())
                total_present = sum(outcome.present_counts.values())
                detail = f"{total_masked} / {total_present}"
            lines.append(f"{outcome.site:<15} {outcome.kind:<5} {outcome.mechanism.value:<18} {detail}")
        census = self.mechanism_census()
        lines.append("")
        lines.append(
            "census: {} sapfluxnet_flags, {} qc_columns, {} none, {} skipped ({} reported)".format(
                census["sapfluxnet_flags"],
                census["qc_columns"],
                census["none"],
                census["skipped"],
                len(self._outcomes),
            )
        )
        unaccounted = self.unaccounted_sites()
        if unaccounted:
            lines.append("UNACCOUNTED: {}".format(", ".join(unaccounted)))
        return "\n".join(lines)
