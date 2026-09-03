"""The Delta writer (IP03) — the only part of the lab that needs a JVM.

Everything else in this repository reads Delta with ``deltalake``, which has no
JVM at all. Writing is different: a MERGE with schema enforcement and a
transaction log entry is Spark's job, and this module is the client that asks
Spark to do it.

It talks to a **Spark Connect** server rather than running ``spark-submit``.
That choice buys three things the lab actually needs: the Airflow image stays a
Python image (no 400 MB of Scala in the scheduler), the driver is a long-lived
service a student can restart independently when it misbehaves, and the same
code runs from a laptop shell as from an Airflow task — the only difference is
``LAB28_SPARK_REMOTE``.

The SQL is not written here. ``lab28_platform.delta_store`` owns the schema, the
``CREATE TABLE`` and the ``MERGE``, because those have to agree with the row
shapers sitting beside them; this module only supplies a session, a DataFrame
and a span.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from opentelemetry.trace import SpanKind
from pyspark.sql import DataFrame, SparkSession

from lab28_platform import metrics
from lab28_platform.contracts import IngestionEvent
from lab28_platform.delta_store import (
    TABLE_SCHEMAS,
    column_names,
    create_table_ddl,
    document_rows,
    feedback_rows,
    merge_sql,
    schema_ddl,
)
from lab28_platform.telemetry import (
    SPAN_SPARK_DELTA_MERGE,
    context_from_traceparent,
    span,
)

logger = logging.getLogger(__name__)

#: The row shaper for each table, keyed the same way as ``TABLE_SCHEMAS`` so a
#: new table cannot be half-added: declare a schema without a shaper and the
#: assertion below fails at import time rather than at 2am in an Airflow log.
ROW_BUILDERS = {
    "feedback": feedback_rows,
    "documents": document_rows,
}
assert set(ROW_BUILDERS) == set(TABLE_SCHEMAS), "every declared table needs a row builder"


class MergeFailed(RuntimeError):
    """The Spark session or the MERGE itself could not complete."""


@dataclass(frozen=True)
class MergeResult:
    """What one batch did to the lakehouse, per table."""

    rows: dict[str, int]
    versions: dict[str, int]
    idempotency_keys: list[str]
    entity_ids: list[str]

    @property
    def total_rows(self) -> int:
        return sum(self.rows.values())

    def version_of(self, table: str) -> int:
        """The version to stamp into evidence for ``table``.

        ``-1`` means "the batch had nothing for this table", which is different
        from version 0 and has to stay different: a downstream consumer that
        treats them alike would pin evidence to a table it never wrote.
        """
        return self.versions.get(table, -1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": dict(self.rows),
            "versions": dict(self.versions),
            "idempotency_keys": list(self.idempotency_keys),
            "entity_ids": list(self.entity_ids),
        }


def connect(remote: str, *, app_name: str = "lab28-delta-merge") -> SparkSession:
    """Open a Spark Connect session, or say plainly that the server is down.

    The endpoint is passed in rather than read from the environment here:
    ``lab28_platform.settings`` is the only module that reads configuration, so
    there is exactly one place to look when a job connects somewhere unexpected.

    The session timezone is pinned to UTC. Without it the same ``occurred_at``
    read back through ``deltalake`` on a laptop in Asia/Ho_Chi_Minh differs from
    what Spark wrote by seven hours, and the freshness metric goes negative.
    """
    try:
        return (
            SparkSession.builder.remote(remote)
            .appName(app_name)
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
    except Exception as error:  # pragma: no cover - needs a live server
        raise MergeFailed(f"cannot reach the Spark Connect server at {remote}: {error}") from error


def ensure_tables(spark: SparkSession, tables: dict[str, str]) -> None:
    """Create every table that does not exist yet, at its configured path."""
    for table, location in tables.items():
        spark.sql(create_table_ddl(table, location))


def _frame(spark: SparkSession, table: str, rows: Sequence[dict[str, Any]]) -> DataFrame:
    """Build the merge source with an explicit schema.

    Explicit, never inferred. Inference on a batch where every ``label`` happens
    to be null types that column as ``VOID`` and the MERGE fails; on a batch
    where every ``rating`` is an ``int`` it picks ``BIGINT`` and the MERGE fails
    differently. Declaring the schema makes both of those impossible.
    """
    order = column_names(TABLE_SCHEMAS[table])
    return spark.createDataFrame(
        [tuple(row[name] for name in order) for row in rows],
        schema=schema_ddl(table),
    )


def _last_commit(spark: SparkSession, location: str) -> tuple[int, dict[str, str]]:
    """The version and operation metrics of the newest commit on a table."""
    history = spark.sql(f"DESCRIBE HISTORY delta.`{location}` LIMIT 1").collect()
    if not history:  # pragma: no cover - a Delta table always has a commit
        return 0, {}
    entry = history[0]
    return int(entry["version"]), dict(entry["operationMetrics"] or {})


def merge_table(
    spark: SparkSession, table: str, location: str, rows: Sequence[dict[str, Any]]
) -> int:
    """MERGE one batch into one table and return the resulting version.

    The insert/update split is read back out of the transaction log rather than
    assumed from the batch size. That is the difference between a metric that
    reports what the writer intended and one that reports what Delta did — and
    on a replay those two disagree, which is exactly the case worth seeing.
    """
    view = f"lab28_merge_source_{table}"
    _frame(spark, table, rows).createOrReplaceTempView(view)
    try:
        spark.sql(merge_sql(location, view))
    except Exception as error:
        raise MergeFailed(f"MERGE into {table} at {location} failed: {error}") from error
    finally:
        spark.catalog.dropTempView(view)

    version, operation_metrics = _last_commit(spark, location)
    for operation, key in (
        ("inserted", "numTargetRowsInserted"),
        ("updated", "numTargetRowsUpdated"),
    ):
        written = int(operation_metrics.get(key, 0))
        if written:
            metrics.DELTA_ROWS_WRITTEN.labels(table=table, operation=operation).inc(written)
    metrics.DELTA_VERSION.labels(table=table).set(version)
    return version


def merge_events(
    spark: SparkSession,
    tables: dict[str, str],
    events: Iterable[IngestionEvent],
    *,
    traceparent: str | None = None,
) -> MergeResult:
    """Write one deduplicated batch of events into the lakehouse.

    The span is a child of the *caller's* trace, not a new one: the whole point
    of IP10 is that the person reading the trace can see the Kafka hop and this
    write as one story. ``traceparent`` arrives from the Airflow DAG conf, which
    got it from the Kafka header, which got it from the HTTP request.
    """
    batch = list(events)
    rows = {table: builder(batch) for table, builder in ROW_BUILDERS.items()}
    written = {table: len(table_rows) for table, table_rows in rows.items()}

    parent = context_from_traceparent(traceparent)
    with span(
        SPAN_SPARK_DELTA_MERGE,
        kind=SpanKind.CLIENT,
        parent=parent,
        service_name="lab28-spark",
        attributes={
            "lab28.batch.events": len(batch),
            "lab28.batch.feedback_rows": written["feedback"],
            "lab28.batch.document_rows": written["documents"],
        },
    ) as active:
        ensure_tables(spark, tables)

        versions: dict[str, int] = {}
        for table, table_rows in rows.items():
            if not table_rows:
                # Merging nothing would still commit a version and make the
                # history unreadable as evidence of what actually arrived.
                continue
            with metrics.DELTA_MERGE_SECONDS.labels(table=table).time():
                versions[table] = merge_table(spark, table, tables[table], table_rows)

        for table, version in versions.items():
            active.set_attribute(f"lab28.delta.{table}_version", version)

    result = MergeResult(
        rows=written,
        versions=versions,
        # Sorted so a replay of the same batch produces a byte-identical
        # ProcessedBatchEvent, which is what makes IT-J2 checkable.
        idempotency_keys=sorted({event.idempotency_key for event in batch}),
        entity_ids=sorted({event.entity_id for event in batch}),
    )
    logger.info(
        "merged %s events into Delta: %s (versions %s)",
        len(batch),
        written,
        versions,
    )
    return result
