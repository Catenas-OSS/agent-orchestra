export class Server {
  constructor(info) { this.info = info; this.tools = new Map(); }
  tool(def) { this.tools.set(def.name, def); }
  async handleRequest(req, ctx) {
    const t = this.tools.get(req.method);
    if (!t) throw new Error(`Unknown method ${req.method}`);
    return t.handler(ctx, req.params);
  }
  async connect(transport) { await transport.start(this); }
}
export class StdioServerTransport { async start() {} }
