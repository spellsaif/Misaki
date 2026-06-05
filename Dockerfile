# Stage 1: Build
FROM rust:1.85-slim-bookworm AS builder

# Create a dummy project to cache dependencies
WORKDIR /usr/src/misaki
COPY Cargo.toml Cargo.lock ./
COPY crates/misaki-core/Cargo.toml crates/misaki-core/
COPY crates/misaki-proxy/Cargo.toml crates/misaki-proxy/

# Create dummy source files for dependency caching
RUN mkdir -p crates/misaki-core/src && \
    echo "" > crates/misaki-core/src/lib.rs && \
    mkdir -p crates/misaki-proxy/src && \
    echo "fn main() {}" > crates/misaki-proxy/src/main.rs && \
    cargo build --release

# Copy the actual source code
COPY crates crates
# Touch files to force rebuild of the crates
RUN touch crates/misaki-core/src/lib.rs && \
    touch crates/misaki-proxy/src/main.rs && \
    cargo build --release

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install CA certificates for outgoing HTTPS requests
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/src/misaki/target/release/misaki-proxy /app/misaki-proxy
COPY misaki.yaml /app/misaki.yaml

EXPOSE 8080
ENV MISAKI_CONFIG=/app/misaki.yaml

CMD ["/app/misaki-proxy"]
