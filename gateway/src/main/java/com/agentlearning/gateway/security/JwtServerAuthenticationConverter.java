package com.agentlearning.gateway.security;

import org.springframework.http.HttpHeaders;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.web.server.authentication.ServerAuthenticationConverter;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

/**
 * 从请求中提取 JWT Token (Spring Security 标准接口)
 * 
 * 标准流程:
 *   1. ServerAuthenticationConverter — 从 HTTP 请求提取凭证 (本类)
 *   2. ReactiveAuthenticationManager — 验证凭证 (JwtAuthenticationManager)
 *   3. 两者由 AuthenticationWebFilter 编排
 * 
 * 支持格式: Authorization: Bearer <token>
 */
@Component
public class JwtServerAuthenticationConverter implements ServerAuthenticationConverter {

    private static final String BEARER_PREFIX = "Bearer ";

    @Override
    public Mono<Authentication> convert(ServerWebExchange exchange) {
        String authHeader = exchange.getRequest().getHeaders().getFirst(HttpHeaders.AUTHORIZATION);

        if (authHeader == null || !authHeader.startsWith(BEARER_PREFIX)) {
            return Mono.empty();
        }

        String token = authHeader.substring(BEARER_PREFIX.length());
        var auth = new UsernamePasswordAuthenticationToken(token, token);
        return Mono.just(auth);
    }
}
