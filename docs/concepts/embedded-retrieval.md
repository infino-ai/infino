# Embedded retrieval

**Embedded retrieval means the search engine runs *in your process*, as a
library, with no server to deploy — the way SQLite is a database you link
against instead of a database you connect to over a socket.** You add Infino as
a dependency, open a connection to a storage root, and query; the engine,
including SQL, full-text, and vector search, executes inside your application.

## Why embed instead of run a server

- **No operations.** There is no daemon to provision, scale, secure, or keep
  warm between queries. The engine's lifecycle is your application's lifecycle.
- **Agents can stand it up unattended.** An AI coding agent (or the code it
  writes) can add the dependency, open a connection, and verify a query in one
  session — no account, API key, or running service required. Hosted-only
  engines can't be brought up this way in a sandbox.
- **Lower latency floor.** A query is a function call, not a network round
  trip to a separate tier.
- **Simple to ship.** One dependency, in three languages — a Rust crate
  (`cargo add infino`), a Python wheel (`pip install infino`), and a Node
  package (`npm install @infino-ai/infino`) — so the same engine embeds
  wherever your application runs.

## How Infino does it

Infino is embedded by design: `connect(uri)` opens an in-process catalog and
everything runs locally.

```python
import infino

# In-process. "memory://" is ephemeral; a path or "s3://bucket/prefix"
# persists. No server is started, here or ever.
db = infino.connect("memory://")
```

What makes the embedded model work at scale is that **state lives in object
storage, not in the process**: the data and indexes are files on S3, Azure, or
local disk, so the in-process engine is stateless and any process can open the
same table. Multiple processes or hosts coordinate through storage — a guarded
manifest pointer — rather than through a shared service. (One consequence: there
is no wire protocol, so external clients can't attach over the network; you
reach Infino from your own code.)

## When it fits — and when it doesn't

Embedding fits applications and agents that want retrieval *inside* the process
over data they already keep in object storage — RAG, agent memory, in-app
search. If you specifically need a long-running network service that many
external clients connect to over a socket, that's the server model Infino
deliberately isn't. See [Tradeoffs and limits](../tradeoffs.md).

## See also

- [Object-storage-native retrieval](object-storage-native-retrieval.md) — why
  state lives in object storage, which is what makes embedding scale.
- [Retrieval for agents](retrieval-for-agents.md) — the workload embedding
  serves best.
- [Architecture overview → Opening Infino](../architecture/overview.md#opening-infino)
  and the [FAQ](../faq.md).
