import Fastify, { FastifyInstance } from "fastify";
import { Server, ServerTransport } from "@modelcontextprotocol/typescript-sdk";

interface Options {
  port: number;
}

export class HttpServerTransport implements ServerTransport {
  private app: FastifyInstance;
  private opts: Options;
  constructor(opts: Options) {
    this.opts = opts;
    this.app = Fastify({ logger: false });
  }
  async start(server: Server): Promise<void> {
    this.app.post("/mcp/", async (request, reply) => {
      const body: any = request.body;
      const ctx = { headers: request.headers };
      try {
        const result = await server.handleRequest(body, ctx);
        reply.send(result);
      } catch (err: any) {
        reply.status(500).send({ error: err.message });
      }
    });
    await this.app.listen({ port: this.opts.port, host: "0.0.0.0" });
  }
  async close() {
    await this.app.close();
  }
}
