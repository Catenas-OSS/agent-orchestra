export type ToolHandler = (ctx: any, input: any) => Promise<any>;

interface ToolDef {
  name: string;
  description?: string;
  inputSchema?: any;
  outputSchema?: any;
  handler: ToolHandler;
}

export class Server {
  tools: Map<string, ToolDef> = new Map();
  info: { name: string; version: string };
  constructor(info: { name: string; version: string }) {
    this.info = info;
  }
  tool(def: ToolDef) {
    this.tools.set(def.name, def);
  }
  async handleRequest(req: { method: string; params: any }, ctx: any) {
    const tool = this.tools.get(req.method);
    if (!tool) throw new Error(`Unknown method ${req.method}`);
    return tool.handler(ctx, req.params);
  }
  async connect(transport: ServerTransport) {
    await transport.start(this);
  }
}

export interface ServerTransport {
  start(server: Server): Promise<void>;
  close?(): Promise<void>;
}

export class StdioServerTransport implements ServerTransport {
  async start(server: Server): Promise<void> {
    // minimal stub: read from stdin JSON-RPC line by line (not implemented)
    // for build only; no runtime implementation
  }
}
