# Memory (Milvus) Plugin for OpenClaw

Long-term memory with vector search for AI conversations, backed by [Milvus](https://milvus.io/). This plugin provides seamless auto-recall and auto-capture of important information from your conversations.

## Features

- **Vector Search**: Semantic search through conversation history using OpenAI embeddings
- **Auto-Recall**: Automatically inject relevant memories into context
- **Auto-Capture**: Automatically save important information from conversations
- **Multi-Client Support**: Share memories across multiple OpenClaw instances using a centralized Milvus server
- **Dual Embedding Modes**: Support for both OpenAI SDK and direct REST API requests

## Prerequisites

- A running Milvus server (local or remote)
- OpenAI API key (for embeddings)

### Quick Start with Milvus

#### Using Docker (Recommended)
```bash
# Start Milvus standalone
docker run -d \
  --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

#### Using Milvus Operator
For Kubernetes deployments, see the [Milvus Operator documentation](https://milvus.io/docs/install_standalone-operator.md).

#### Using Zilliz Cloud
Sign up for [Zilliz Cloud](https://cloud.zilliz.com/) for a managed Milvus service.

## Installation

1. Enable the plugin in your OpenClaw configuration
2. Configure your Milvus connection and OpenAI API key

## Configuration

Add this to your OpenClaw settings:

```json
{
  "plugins": {
    "memory-milvus": {
      "embedding": {
        "apiKey": "${OPENAI_API_KEY}",
        "model": "text-embedding-3-small",
        "clientType": "openapi-client",
        "baseUrl": "https://api.openai.com/v1",
        "dimensions": 1536
      },
      "milvus": {
        "address": "localhost:19530",
        "collectionName": "openclaw_memories",
        "secure": false,
        "username": "",
        "password": "",
        "token": ""
      },
      "autoRecall": true,
      "autoCapture": true,
      "captureMaxChars": 500
    }
  }
}
```

### Configuration Options

#### Embedding Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `apiKey` | string | **required** | OpenAI API key (can use `${OPENAI_API_KEY}` env var) |
| `model` | string | `text-embedding-3-small` | Embedding model to use |
| `clientType` | string | `openapi-client` | Client type: `openapi-client` (OpenAI SDK) or `rest` (direct HTTP) |
| `baseUrl` | string | `https://api.openai.com/v1` | Base URL for compatible embedding providers |
| `dimensions` | number | (model-specific) | Vector dimensions (required for custom models) |

#### Milvus Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `address` | string | **required** | Milvus server address (host:port) |
| `collectionName` | string | `openclaw_memories` | Collection name for storing memories |
| `secure` | boolean | `false` | Enable TLS/SSL for Milvus connection |
| `username` | string | (optional) | Username for Milvus authentication |
| `password` | string | (optional) | Password for Milvus authentication |
| `token` | string | (optional) | Token for Milvus authentication (alternative to username/password) |

#### Memory Behavior Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `autoRecall` | boolean | `true` | Automatically inject relevant memories into context |
| `autoCapture` | boolean | `false` | Automatically capture important information |
| `captureMaxChars` | number | `500` | Maximum message length eligible for auto-capture |

## Usage

### Tools

#### `memory_recall`
Search through long-term memories. Use when you need context about user preferences, past decisions, or previously discussed topics.

**Parameters:**
- `query` (string, required): Search query
- `limit` (number, optional): Max results (default: 5)

#### `memory_store`
Save important information in long-term memory. Use for preferences, facts, decisions.

**Parameters:**
- `text` (string, required): Information to remember
- `importance` (number, optional): Importance 0-1 (default: 0.7)
- `category` (string, optional): Memory category (preference, fact, decision, entity, other)

#### `memory_forget`
Delete specific memories. GDPR-compliant.

**Parameters:**
- `query` (string, optional): Search to find memory
- `memoryId` (string, optional): Specific memory ID

### CLI Commands

```bash
# List memory count
openclaw ltm list

# Search memories
openclaw ltm search "my preferences" --limit 10

# Show memory statistics
openclaw ltm stats
```

## Memory Categories

Memories are automatically categorized into:
- `preference` - User likes, dislikes, preferences
- `fact` - Factual information
- `decision` - Decisions made
- `entity` - People, places, contact info
- `other` - Everything else

## Multi-Client Setup

To share memories across multiple OpenClaw clients:

1. Set up a centralized Milvus server (or use Zilliz Cloud)
2. Configure all clients to use the same Milvus `address` and `collectionName`
3. Memories will be automatically shared and synchronized across all connected clients

## Security Considerations

- Always use environment variables for sensitive credentials (e.g., `${OPENAI_API_KEY}`)
- Enable TLS (`secure: true`) when connecting to remote Milvus servers
- Consider network-level security (VPN, private networks) for production deployments
- Use Milvus authentication in production environments

## Troubleshooting

### Connection Issues
- Verify Milvus is running: `curl http://localhost:9091/healthz`
- Check firewall settings for port 19530
- Try `secure: false` first when testing locally

### Collection Creation Errors
- Ensure your Milvus user has sufficient permissions
- Check if the collection already exists with a different schema

### Embedding Issues
- Verify your OpenAI API key is valid
- Check your API quota and billing status
- Try switching `clientType` between `openapi-client` and `rest`

## Contributing

This plugin is part of the OpenClaw project. Contributions are welcome!

## License

See the OpenClaw project license for details.