import { Server } from "@modelcontextprotocol/typescript-sdk";
import { registerAllTools } from "./tools/registry.js";
import pkg from "../package.json" assert { type: "json" };

export async function createServer() {
  const server = new Server({
    name: "slack",
    version: pkg.version,
  });
  await registerAllTools(server);
  return server;
}
