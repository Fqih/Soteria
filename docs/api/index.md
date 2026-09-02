# API reference

The public API surface is generated from inline docstrings via
[`mkdocstrings`](https://mkdocstrings.github.io/). Modules below are
considered **stable** — see [API stability](../api-stability.md) for the
deprecation policy.

## Core runtime

::: avo.runtime.AgentRuntime
    options:
      members:
        - run
        - resume

## Models

::: avo.models.ModelRequest

::: avo.models.ModelResponse

## Loop policy

::: avo.policies.LoopPolicy

## Providers

::: avo.providers.base.ModelProvider

## Tools

::: avo.tools.FunctionTool

::: avo.tools.ToolRegistry

## Circuit breaker

::: avo.circuit_breaker.CircuitBreaker

::: avo.circuit_breaker.CircuitBreakerPolicy

## Logging

::: avo.logging_config.configure_logging
