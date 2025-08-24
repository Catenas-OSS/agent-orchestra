# @modelcontextprotocol/server-slack

This package implements a Model Context Protocol (MCP) server that exposes Slack Web API methods as MCP tools.

## Usage

### STDIO

```bash
SLACK_TOKEN=xoxb-... npx @modelcontextprotocol/server-slack --transport=stdio
```

### HTTP

```bash
SLACK_TOKEN=xoxb-... npx @modelcontextprotocol/server-slack --transport=http --port=4000
```

Then configure your MCP client to send requests to `http://localhost:4000/mcp/` with an `Authorization: Bearer <token>` header.

### Tools

Generated tools follow the pattern `slack.<namespace>.<method>` such as `slack.chat.postMessage`. A generic `slack.call` tool is also available.

### Code Generation

Run `npm run codegen` to regenerate all Slack method tools using Slack's OpenAPI spec or a static fallback.

## Development

- `npm run dev` – run in development mode.
- `npm run build` – compile to `dist/`.
- `npm test` – run tests.

## License

MIT
