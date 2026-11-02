"""Which sites a product legitimately cannot cover, and the reason for each.

An allowlist, **not** a lowered threshold.  The coverage check exists to catch
an extraction that silently produced nothing; relaxing it to make the gate go
green would throw that away.  So each site a product genuinely cannot cover is
named here together with why, and three outcomes stay distinguishable:

* **explained**   -- absent, and the reason is recorded below
* **unexplained** -- absent for no recorded reason; still a hard failure
* **stale**       -- listed here but present after all, so this file is wrong

The third case matters as much as the first.  Without it an allowlist entry
outlives the condition that justified it and quietly excuses a real regression
years later.

GlobMap LAI V3 is the only current case.  The rasters on disk run
``GlobMapLAIV3.A1991001`` .. ``GlobMapLAIV3.A2020361`` -- 1991-01-01 to
2020-12-26 -- while the combined corpus reaches 2025-12-11.  Four sites begin
after the last raster, so there is nothing to sample for any of their hours.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

GLOBMAP_LAI_LAST_RASTER = "2020-12-26"
"""Date of GlobMapLAIV3.A2020361, the newest LAI raster held locally."""

KNOWN_ABSENT: dict[str, dict[str, str]] = {
    "globmap_lai": {
        "AUT_MMG": f"record starts 2023-02-15, after GlobMap LAI V3 ends {GLOBMAP_LAI_LAST_RASTER}",
        "DEU_HAI": f"record starts 2022-06-18, after GlobMap LAI V3 ends {GLOBMAP_LAI_LAST_RASTER}",
        "DEU_HAR": f"record starts 2023-04-27, after GlobMap LAI V3 ends {GLOBMAP_LAI_LAST_RASTER}",
        "NOR_HUR": f"record starts 2022-05-11, after GlobMap LAI V3 ends {GLOBMAP_LAI_LAST_RASTER}",
    },
}


@dataclass(frozen=True)
class MissingReport:
    """How a product's missing sites divide into excused and not."""

    product: str
    explained: tuple[tuple[str, str], ...] = ()
    unexplained: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when nothing is wrong: every absence is accounted for."""
        return not self.unexplained and not self.stale

    def lines(self) -> list[str]:
        """Human-readable findings, one per line, empty when fully explained."""
        out = []
        for site, reason in self.explained:
            out.append(f"known-absent {site}: {reason}")
        if self.unexplained:
            out.append(f"{len(self.unexplained)} site(s) missing with no recorded reason: {list(self.unexplained)}")
        if self.stale:
            out.append(
                f"{len(self.stale)} allowlist entr(y/ies) for site(s) that are present, "
                f"so this allowlist is out of date: {list(self.stale)}"
            )
        return out

    def problem(self) -> str | None:
        """One-line failure summary, or None when nothing is wrong."""
        if self.ok:
            return None
        parts = []
        if self.unexplained:
            parts.append(f"{len(self.unexplained)} unexplained absence(s)")
        if self.stale:
            parts.append(f"{len(self.stale)} stale allowlist entr(y/ies)")
        return f"{self.product}: " + ", ".join(parts)


def classify_missing(
    product: str,
    missing: Iterable[str],
    present: Iterable[str] = (),
) -> MissingReport:
    """Split ``missing`` into explained and unexplained for ``product``.

    ``present`` is the set of sites the product does cover; supplying it lets
    the allowlist be checked against reality rather than trusted.  A product
    with no allowlist entry treats every absence as unexplained, which is the
    safe default for a product nobody has reasoned about yet.
    """
    allowed = KNOWN_ABSENT.get(product, {})
    missing_set = set(missing)
    present_set = set(present)

    explained = tuple((site, allowed[site]) for site in sorted(missing_set) if site in allowed)
    unexplained = tuple(site for site in sorted(missing_set) if site not in allowed)
    stale = tuple(sorted(set(allowed) & present_set))

    return MissingReport(
        product=product,
        explained=explained,
        unexplained=unexplained,
        stale=stale,
    )
