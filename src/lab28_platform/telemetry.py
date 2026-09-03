"""OpenTelemetry setup and W3C context propagation.

OpenTelemetry is the cross-service telemetry contract for this platform. Every
service exports OTLP to the collector, and the collector fans the same stream
out to a local trace backend and — when a credential is supplied — to LangSmith.

The span names emitted here are the ones asserted by the live trace-continuity
gate; they are listed in ``contracts/integration-matrix.yaml`` under
``required_spans``. Renaming a span is a contract change.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from typing import Any

from opentelemetry import metrics, propagate, trace
from opentelemetry.context import Context
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from lab28_platform.settings import TelemetrySettings

SERVICE_TRACER_NAME = "lab28-platform"

# Span names that make up the required end-to-end trace.
SPAN_GATEWAY_REQUEST = "lab28.gateway.request"
SPAN_API_INGEST = "lab28.api.ingest"
SPAN_API_ASK = "lab28.api.ask"
SPAN_KAFKA_PRODUCE = "lab28.kafka.produce"
SPAN_KAFKA_CONSUME = "lab28.kafka.consume"
SPAN_AIRFLOW_DAG = "lab28.airflow.dag"
SPAN_SPARK_DELTA_MERGE = "lab28.spark.delta_merge"
SPAN_FEAST_LOOKUP = "lab28.feast.get_online_features"
SPAN_QDRANT_QUERY = "lab28.qdrant.query"
SPAN_MLFLOW_RESOLVE = "lab28.mlflow.resolve_release"
SPAN_VLLM_CHAT = "lab28.vllm.chat_completion"
SPAN_GUARDRAIL = "lab28.guardrail.check"

_TRACEPARENT_HEADER = "traceparent"
_TRACESTATE_HEADER = "tracestate"

_configured = False
_service_tracers: dict[str, trace.Tracer] = {}


def configure_telemetry(settings: TelemetrySettings | None = None) -> None:
    """Install the OTLP tracer and meter providers exactly once.

    Safe to call from every entry point (API, CLI, Airflow task, Spark driver).
    When OTLP export is disabled the process still produces spans and contexts,
    so trace propagation keeps working in unit tests without a collector.
    """
    global _configured
    if _configured:
        return
    _configured = True

    active = settings or TelemetrySettings.from_env()

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    resource = Resource.create(
        {
            "service.name": active.service_name,
            "service.namespace": "lab28",
            "deployment.environment": "lab",
        }
    )
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(active.sample_ratio)),
    )

    if active.enabled:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=active.otlp_endpoint, insecure=True))
        )
    if active.console_export:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)

    if active.enabled:
        _configure_metrics(resource, active)


def _configure_metrics(resource: Any, settings: TelemetrySettings) -> None:
    """Export OTLP metrics alongside the Prometheus scrape endpoint."""
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=settings.otlp_endpoint, insecure=True),
        export_interval_millis=10_000,
    )
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))


def get_tracer() -> trace.Tracer:
    configure_telemetry()
    return trace.get_tracer(SERVICE_TRACER_NAME)


def get_service_tracer(service_name: str) -> trace.Tracer:
    """Return a tracer with a distinct service resource for a remote boundary.

    Spark Connect executes the plan in another process even though its Python
    client lives in the Airflow task. Giving that client span its own resource
    keeps the process boundary visible in the distributed trace.
    """
    cached = _service_tracers.get(service_name)
    if cached is not None:
        return cached

    active = TelemetrySettings.from_env(service_name)
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": service_name,
                "service.namespace": "lab28",
                "deployment.environment": "lab",
            }
        )
    )
    if active.enabled:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=active.otlp_endpoint, insecure=True))
        )
    if active.console_export:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    tracer = provider.get_tracer(SERVICE_TRACER_NAME)
    _service_tracers[service_name] = tracer
    return tracer


def trace_id_of(span: Span) -> str:
    """32-hex-character W3C trace ID for the span."""
    return f"{span.get_span_context().trace_id:032x}"


def current_trace_id() -> str:
    return trace_id_of(trace.get_current_span())


def current_traceparent() -> str | None:
    """Serialise the active context into a W3C ``traceparent`` string."""
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier.get(_TRACEPARENT_HEADER)


def context_from_traceparent(traceparent: str | None) -> Context | None:
    """Rebuild a context from an inbound ``traceparent`` header."""
    if not traceparent:
        return None
    return propagate.extract({_TRACEPARENT_HEADER: traceparent})


# --------------------------------------------------------------------------
# Kafka header propagation
# --------------------------------------------------------------------------


def inject_kafka_headers(extra: dict[str, str] | None = None) -> list[tuple[str, bytes]]:
    """Build confluent-kafka message headers carrying the active trace context.

    Kafka headers are the transport for W3C context across the asynchronous
    boundary. Without this the consumer starts a brand new trace and the
    end-to-end trace gate fails.
    """
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    if extra:
        carrier.update(extra)
    return [(key, value.encode("utf-8")) for key, value in carrier.items()]


def context_from_kafka_headers(
    headers: Sequence[tuple[str, bytes | None]] | None,
) -> Context | None:
    """Extract the parent context from inbound Kafka message headers."""
    if not headers:
        return None
    carrier = {
        key: value.decode("utf-8")
        for key, value in headers
        if value is not None and key in {_TRACEPARENT_HEADER, _TRACESTATE_HEADER}
    }
    if _TRACEPARENT_HEADER not in carrier:
        return None
    return propagate.extract(carrier)


def traceparent_from_kafka_headers(
    headers: Sequence[tuple[str, bytes | None]] | None,
) -> str | None:
    for key, value in headers or []:
        if key == _TRACEPARENT_HEADER and value is not None:
            return value.decode("utf-8")
    return None


# --------------------------------------------------------------------------
# Span helpers
# --------------------------------------------------------------------------


@contextmanager
def span(
    name: str,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    parent: Context | None = None,
    attributes: dict[str, Any] | None = None,
    service_name: str | None = None,
) -> Iterable[Span]:
    """Start a span, record exceptions and always set an explicit status."""
    tracer = get_service_tracer(service_name) if service_name else get_tracer()
    with tracer.start_as_current_span(
        name, context=parent, kind=kind, attributes=attributes or {}
    ) as active:
        try:
            yield active
        except Exception as error:
            active.record_exception(error)
            active.set_status(Status(StatusCode.ERROR, str(error)))
            raise
        else:
            # ``NonRecordingSpan`` intentionally exposes no readable ``status``
            # property. That is the span returned when telemetry is disabled,
            # which is a supported laptop mode. ``set_status`` is part of the
            # common Span API and is a harmless no-op in that case.
            active.set_status(Status(StatusCode.OK))


def instrument_fastapi(app: Any) -> None:
    """Attach FastAPI auto-instrumentation when the package is installed."""
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:  # pragma: no cover - optional dependency
        return
    FastAPIInstrumentor.instrument_app(app, excluded_urls="/health,/metrics,/startup")


def instrument_httpx() -> None:
    """Attach httpx auto-instrumentation so outbound calls join the trace."""
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    except ImportError:  # pragma: no cover - optional dependency
        return
    HTTPXClientInstrumentor().instrument()
