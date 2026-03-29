package com.agentlearning.gateway.config;

import com.agentlearning.gateway.security.JwtAuthenticationManager;
import com.agentlearning.gateway.security.JwtServerAuthenticationConverter;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.reactive.EnableWebFluxSecurity;
import org.springframework.security.config.web.server.SecurityWebFiltersOrder;
import org.springframework.security.config.web.server.ServerHttpSecurity;
import org.springframework.security.web.server.SecurityWebFilterChain;
import org.springframework.security.web.server.authentication.AuthenticationWebFilter;

/**
 * Spring Security 配置 (Reactive)
 * 
 * 标准做法:
 *   - 使用 SecurityWebFilterChain 而非手写 GlobalFilter
 *   - AuthenticationWebFilter + ReactiveAuthenticationManager
 *   - 白名单路径通过 permitAll() 配置
 *   - 当 auth.enabled=false 时，全部 permitAll
 */
@Configuration
@EnableWebFluxSecurity
@RequiredArgsConstructor
public class SecurityConfig {

    private final AuthProperties authProperties;
    private final JwtAuthenticationManager authenticationManager;
    private final JwtServerAuthenticationConverter authenticationConverter;

    @Bean
    public SecurityWebFilterChain securityFilterChain(ServerHttpSecurity http) {
        // CSRF 对 API Gateway 无意义 (无 session)
        http.csrf(ServerHttpSecurity.CsrfSpec::disable);

        if (!authProperties.isEnabled()) {
            // 认证关闭 → 全部放行
            http.authorizeExchange(exchanges -> exchanges.anyExchange().permitAll());
            return http.build();
        }

        // 认证开启 → 白名单放行，其余需要 JWT
        String[] publicPaths = authProperties.getPublicPaths().toArray(String[]::new);

        http.authorizeExchange(exchanges -> exchanges
                .pathMatchers(publicPaths).permitAll()
                .anyExchange().authenticated()
        );

        // 注入 JWT AuthenticationWebFilter
        var authFilter = new AuthenticationWebFilter(authenticationManager);
        authFilter.setServerAuthenticationConverter(authenticationConverter);
        http.addFilterAt(authFilter, SecurityWebFiltersOrder.AUTHENTICATION);

        return http.build();
    }
}
