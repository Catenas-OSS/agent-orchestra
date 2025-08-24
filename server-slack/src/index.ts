import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import { createServer } from "./server.js";
import { StdioServerTransport } from "./transport/stdio.js";
import { HttpServerTransport } from "./transport/http.js";

const argv = yargs(hideBin(process.argv))
  .option("transport", {
    type: "string",
    choices: ["stdio", "http"],
    default: "stdio",
  })
  .option("port", { type: "number", default: 4000 })
  .parseSync();

(async () => {
  const server = await createServer();
  if (argv.transport === "http") {
    const transport = new HttpServerTransport({ port: argv.port });
    await server.connect(transport);
  } else {
    const transport = new StdioServerTransport();
    await server.connect(transport);
  }
})();
