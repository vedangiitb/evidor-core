# CHANGELOG

<!-- version list -->

## v1.4.2 (2026-10-10)


## v1.4.2-dev.1 (2026-10-10)

### Bug Fixes

- **mcp**: Adding mcp configs property on mcp client
  ([`5bc87e1`](https://github.com/vedangiitb/evidor-core/commit/5bc87e163488b3d87c5d2a705039f7dfcc3b362b))


## v1.4.1 (2026-10-10)


## v1.4.1-dev.1 (2026-10-10)

### Bug Fixes

- **agent**: Fix summarization dropping info bug
  ([`3223a56`](https://github.com/vedangiitb/evidor-core/commit/3223a565d4b1f39040cf60cfca6c0853c81b3469))


## v1.4.0 (2026-10-10)


## v1.4.0-dev.2 (2026-10-10)

### Bug Fixes

- **agent**: Fixed custom sync sleeper issue
  ([`38aaa8a`](https://github.com/vedangiitb/evidor-core/commit/38aaa8aad45275006702ecb2eb5b7b41f613edfc))

- **agent**: Fixing gemini default retries issue
  ([`c3cd6b9`](https://github.com/vedangiitb/evidor-core/commit/c3cd6b93ff4b2b79587ae35bc03f0eeeb9aee2ce))

- **retry**: Harden llm retry policies, async backoff, and telemetry coordination
  ([`c93a8ef`](https://github.com/vedangiitb/evidor-core/commit/c93a8efd766446a2fce8e90a9a13be1c791ccb26))

### Documentation

- **agent**: Updated readme with agent retries info
  ([`41bbb28`](https://github.com/vedangiitb/evidor-core/commit/41bbb28bb4669500dafd7357dabd0565f491f475))

### Features

- Python 3.14 support
  ([`83cfe8e`](https://github.com/vedangiitb/evidor-core/commit/83cfe8e158a564bd950266b7f265f5d0ba0f55fe))

- **agent**: Added configurable agent retries
  ([`2ed8f12`](https://github.com/vedangiitb/evidor-core/commit/2ed8f12ee8ff8e9c44f58501a898b05c656e7325))


## v1.4.0-dev.1 (2026-10-09)

### Bug Fixes

- **telemetry**: Guarantee queue draining on shutdown and in-flight flush synchronization
  ([`6d20487`](https://github.com/vedangiitb/evidor-core/commit/6d204878a24ac8f52652b4dd077c071d6154f9a8))

### Documentation

- **readme**: Add adapter overview table and optional extras summary
  ([`a70fe7c`](https://github.com/vedangiitb/evidor-core/commit/a70fe7cab3f32064552ff84358b6d06611ccc107))

- **telemetry**: Document Langfuse, Arize Phoenix, and Prometheus adapters
  ([`6b687cc`](https://github.com/vedangiitb/evidor-core/commit/6b687cc0335beda4af2bbd13f9514fd5c5724c46))

- **telemetry**: Document telemetry usage in readme and configure evidor[otel] packaging
  ([`1222de0`](https://github.com/vedangiitb/evidor-core/commit/1222de027ad852842519a8cad4f02af397b91901))

### Features

- **telemetry**: Add Langfuse, Arize Phoenix, and Prometheus adapters
  ([`188fa50`](https://github.com/vedangiitb/evidor-core/commit/188fa50ec176e96e30708f51149e87adf9a79e25))

- **telemetry**: Add sink-level capture_content option for PII protection
  ([`6ae868f`](https://github.com/vedangiitb/evidor-core/commit/6ae868f37c7c3916760bcdd55173b6ebbe800f61))

- **telemetry**: Define domain events, trace context, and semantic conventions
  ([`17b9730`](https://github.com/vedangiitb/evidor-core/commit/17b9730bbc2bce395a21313fd1a4f9d70276f730))

- **telemetry**: Implement actor runtime, core instrumentation, and opentelemetry adapter
  ([`124deb0`](https://github.com/vedangiitb/evidor-core/commit/124deb0531985b1a2a781c395db5db177bd38e33))

### Refactoring

- **telemetry**: Streamline Langfuse and Prometheus adapter implementations
  ([`f1f0eb8`](https://github.com/vedangiitb/evidor-core/commit/f1f0eb854694c54b91c6a1e06fc18e8837bcf6f0))


## v1.3.0 (2026-10-08)


## v1.3.0-dev.2 (2026-10-06)

### Bug Fixes

- Adding parallel connectivity to mcp for sync process
  ([`83a4173`](https://github.com/vedangiitb/evidor-core/commit/83a417343721f6e1a9da73cdb5dd3290b91a7392))


## v1.3.0-dev.1 (2026-10-06)

### Bug Fixes

- Yml updates for fixing ci failure
  ([`e76d123`](https://github.com/vedangiitb/evidor-core/commit/e76d123b3282fd4a3dda51b91ac71dc25e9c50af))

### Chores

- Config and requirement updates for mcp integration
  ([`83e419e`](https://github.com/vedangiitb/evidor-core/commit/83e419e4564e746f16295183179f6661a2fc8978))

### Documentation

- **mcp**: Mcp updates for readme
  ([`83c8f96`](https://github.com/vedangiitb/evidor-core/commit/83c8f96bbc64f4026701ab9f17dc0a2348d9c075))

### Features

- Agent updates for mcp integration
  ([`843ab00`](https://github.com/vedangiitb/evidor-core/commit/843ab00f54f5d95c235f86abbde328bd6523b542))

- **mcp**: Mcp integration
  ([`44ad88d`](https://github.com/vedangiitb/evidor-core/commit/44ad88d1b03e0923fc0f7a47fd1ec9d1ec3ff7c0))

- **mcp**: Tool executor updates for mcp integration
  ([`532c5ac`](https://github.com/vedangiitb/evidor-core/commit/532c5acbb5ddb719b42fbd61bd41f1bb65e929aa))


## v1.2.0 (2026-10-04)


## v1.1.0-dev.3 (2026-10-04)

### Documentation

- **tools**: Doc updates for web search tool
  ([`f07d47f`](https://github.com/vedangiitb/evidor-core/commit/f07d47f7f988d2e0751fd4981ef94160d09f48da))

### Features

- **tools**: Web search tool
  ([`e5f051b`](https://github.com/vedangiitb/evidor-core/commit/e5f051b6b53ba14237f9aab4254715c36a68dcf8))


## v1.1.0-dev.2 (2026-10-04)
## v1.1.0 (2026-10-04)

### Chores

- Readme updates for fs tools
  ([`2da0bfd`](https://github.com/vedangiitb/evidor-core/commit/2da0bfdd1600ddf9ba8ae5eb8c5eb740dbaa7e84))

### Features

- Adding fs tools
  ([`ccc0ef9`](https://github.com/vedangiitb/evidor-core/commit/ccc0ef94e14c0f1dd1f89db14aefd21d1a658a5d))


## v1.1.0-dev.1 (2026-09-30)

### Features

- Adding built in tools
  ([`6fb96c2`](https://github.com/vedangiitb/evidor-core/commit/6fb96c2c83fc50eb51728d7811b671f05ea32382))


## v1.0.0 (2026-09-30)

### Features

- Stable release with tool primitives
  ([`ccb21cb`](https://github.com/vedangiitb/evidor-core/commit/ccb21cbc821b2f65ea8eedfae7de556c8c8f4073))


## v1.0.0-dev.4 (2026-09-29)

### Bug Fixes

- Adding pytest asyncio to fix ci failure
  ([`1c83baa`](https://github.com/vedangiitb/evidor-core/commit/1c83baa0655eea18442492d80c531eca61bb8e50))

### Chores

- Python 3.13 in yml
  ([`d749ca3`](https://github.com/vedangiitb/evidor-core/commit/d749ca30dd5a99b4dc5bc5a0f224efa2370456a4))

### Features

- Async agent and tool executions
  ([`5ad1759`](https://github.com/vedangiitb/evidor-core/commit/5ad1759bd66198db6d34ef7742b300e0ff582cfd))


## v1.0.0-dev.3 (2026-09-28)

### Features

- Introduced tool primitives for integrating tool calls
  ([`9edbf58`](https://github.com/vedangiitb/evidor-core/commit/9edbf58c74b6ea3c9b6a184227353e36a794ad54))


## v1.0.0-dev.2 (2026-09-26)

### Bug Fixes

- Yml changes for adding missing dependencies
  ([`7130909`](https://github.com/vedangiitb/evidor-core/commit/71309090357628c2d9c37977b19d8c9c13cd4882))

### Chores

- Readme updates and adding requirements file
  ([`9e9a28b`](https://github.com/vedangiitb/evidor-core/commit/9e9a28b8e14db5c0a4f0b718fa97c8376f8d5199))

- Updated environment names
  ([`8d4b5ce`](https://github.com/vedangiitb/evidor-core/commit/8d4b5ce10a664ba38673982e20f480ff677f3561))

### Features

- Context memory for longer chats
  ([`6103b4f`](https://github.com/vedangiitb/evidor-core/commit/6103b4f91b83b80cc588c3a61cb5ab28041b21a5))


## v1.0.0-dev.1 (2026-09-25)

### Chores

- Added git yml
  ([`d48c4f4`](https://github.com/vedangiitb/evidor-core/commit/d48c4f491633669908b01b0b5437424c1bc6a4e1))


## v0.1.0 (2026-09-25)

- Initial Release
