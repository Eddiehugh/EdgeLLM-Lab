"""Runtime implementations and factory."""

from execution.runtimes.base import Runtime
from execution.runtimes.docker import DockerRuntime
from execution.runtimes.native import NativeRuntime
from execution.specs import RuntimeSpec


def build_runtime(spec: RuntimeSpec) -> Runtime:
    runtimes = {"native": NativeRuntime, "docker": DockerRuntime}
    try:
        runtime_type = runtimes[spec.type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown runtime '{spec.type}'. Available: {', '.join(runtimes)}"
        ) from exc
    return runtime_type(spec)


__all__ = ["DockerRuntime", "NativeRuntime", "Runtime", "build_runtime"]
