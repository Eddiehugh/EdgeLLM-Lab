"""Machine-readable reports shared by compression transforms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModuleCompressionRecord:
    """Compression measurements for one transformed module."""

    name: str
    module_type: str
    original_bytes: int
    compressed_bytes: int
    parameter_count: int
    affected_parameters: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompressionReport:
    """Aggregate measurements produced by one compression method."""

    method: str
    records: tuple[ModuleCompressionRecord, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    model_original_bytes: int | None = None
    model_compressed_bytes: int | None = None

    @classmethod
    def combine(
        cls,
        method: str,
        reports: Sequence["CompressionReport"],
    ) -> "CompressionReport":
        """Combine disjoint compression passes without losing pass metadata."""

        if len(reports) == 1:
            return reports[0]
        records = tuple(record for report in reports for record in report.records)
        return cls(
            method=method,
            records=records,
            metadata={
                "passes": [
                    {
                        "method": report.method,
                        "module_count": len(report.records),
                        "metadata": dict(report.metadata),
                    }
                    for report in reports
                ]
            },
            model_original_bytes=reports[0].model_original_bytes if reports else None,
            model_compressed_bytes=(
                reports[-1].model_compressed_bytes if reports else None
            ),
        )

    @property
    def original_bytes(self) -> int:
        return sum(record.original_bytes for record in self.records)

    @property
    def compressed_bytes(self) -> int:
        return sum(record.compressed_bytes for record in self.records)

    @property
    def parameter_count(self) -> int:
        return sum(record.parameter_count for record in self.records)

    @property
    def affected_parameters(self) -> int:
        return sum(record.affected_parameters for record in self.records)

    @property
    def compression_ratio(self) -> float:
        if self.compressed_bytes == 0:
            return float("inf") if self.original_bytes else 1.0
        return self.original_bytes / self.compressed_bytes

    @property
    def affected_fraction(self) -> float:
        if self.parameter_count == 0:
            return 0.0
        return self.affected_parameters / self.parameter_count

    @property
    def model_compression_ratio(self) -> float | None:
        if self.model_original_bytes is None or self.model_compressed_bytes is None:
            return None
        if self.model_compressed_bytes == 0:
            return float("inf") if self.model_original_bytes else 1.0
        return self.model_original_bytes / self.model_compressed_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "module_count": len(self.records),
            "original_bytes": self.original_bytes,
            "compressed_bytes": self.compressed_bytes,
            "compression_ratio": self.compression_ratio,
            "parameter_count": self.parameter_count,
            "affected_parameters": self.affected_parameters,
            "affected_fraction": self.affected_fraction,
            "model_original_bytes": self.model_original_bytes,
            "model_compressed_bytes": self.model_compressed_bytes,
            "model_compression_ratio": self.model_compression_ratio,
            "metadata": dict(self.metadata),
            "modules": [record.to_dict() for record in self.records],
        }
