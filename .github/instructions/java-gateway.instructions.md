---
applyTo: "gateway/**"
---
# Java Gateway Instructions

- **Framework**: Spring Cloud Gateway (reactive, WebFlux-based). All handlers return `Mono<>` / `Flux<>`.
- **Build**: `cd gateway && mvn clean package -DskipTests` → `target/gateway-1.0.0.jar`
- **Run**: `java -jar gateway/target/gateway-1.0.0.jar` or `cd gateway && mvn spring-boot:run`
- **Port**: `:8080` (configurable via `SERVER_PORT` env var)
- **Routing**: Defined in `application.yml` under `spring.cloud.gateway.routes`. Pattern: `StripPrefix=2` removes `/api/{service}` prefix before forwarding.
- **Filters execute in order**: JwtAuthFilter (-100) → RequestLoggingFilter (-90) → route-level filters
- **JWT auth is off by default** (`gateway.auth.enabled=false`). When enabled, non-public paths require `Authorization: Bearer <token>`.
- **Fallback**: Each route has a `CircuitBreaker` filter pointing to `/fallback/{service}` — returns friendly JSON, not 502.
- **Add new downstream service**: Add route in `application.yml`, add fallback method in `FallbackController`, add health check in `HealthAggregationController`.
