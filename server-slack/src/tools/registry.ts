import { Server } from "@modelcontextprotocol/typescript-sdk";
import * as genericCall from "./genericCall.js";
import * as chatPostMessage from "./generated/chat.postMessage.js";
import * as usersList from "./generated/users.list.js";

const tools = [genericCall, chatPostMessage, usersList];

export async function registerAllTools(server: Server) {
  for (const t of tools) {
    server.tool({
      name: (t as any).name,
      description: (t as any).description,
      inputSchema: (t as any).inputSchema,
      outputSchema: (t as any).outputSchema,
      handler: (t as any).handler,
    });
  }
}
